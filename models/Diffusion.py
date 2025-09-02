import random
from typing import List
from torch import Tensor
import torch
from torch import  nn
from models.embeddings import (TimestepEmbedding,Timesteps)
# from models.operators import PositionalEncoding
from models.operators.cross_attention import (SkipTransformerEncoder,
                                              TransformerDecoder,
                                              TransformerDecoderLayer,
                                              TransformerEncoder,
                                              TransformerEncoderLayer)
from models.operators.position_encoding import build_position_encoding


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def sample_timesteps(self, n):
    return torch.randint(low=1, high=self.noise_steps, size=(n,))

def add_noise(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 256,
                 condition: str = "text",
                 motion_dim: int = 48,
                    seq_len: int = 125,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.motion_dim = motion_dim
        self.seq_len = seq_len
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE

        if 0:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)



        self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,freq_shift)
        self.time_embedding = TimestepEmbedding(self.latent_dim,self.latent_dim)
        self.emb_proj = EmbedMotion(self.motion_dim,
                                    self.seq_len,
                                    self.latent_dim,
                                    guidance_scale=guidance_scale,
                                    guidance_uncodp=guidance_uncondp)

        #PE实现
        # if self.pe_type == "actor":
        #     self.query_pos = PositionalEncoding(self.latent_dim, dropout)
        #     self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        # elif self.pe_type == "mld":
        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        # else:
        #     raise ValueError("Not Support PE type")


        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    def forward(self,
                sample,
                timestep,
                row_seq,
                lengths=None,
                pre_length=100,
                **kwargs):
        # 0.  dimension matching
        #
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding 同上文本和类型编码不适用，时间编码考虑使用
        # if self.condition in ["text", "text_uncond"]:
        #     # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
        #     encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        #     text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
        #     # textembedding projection
        #     if self.text_encoded_dim != self.latent_dim:
        #         # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
        #         text_emb_latent = self.emb_proj(text_emb)
        #     else:
        #         text_emb_latent = text_emb
        #     if self.abl_plus:
        #         emb_latent = time_emb + text_emb_latent
        #     else:
        #         emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        # elif self.condition in ['action']:
        #     action_emb = self.emb_proj(encoder_hidden_states)
        #     if self.abl_plus:
        #         emb_latent = action_emb + time_emb
        #     else:
        #         emb_latent = torch.cat((time_emb, action_emb), 0)
        # else:
        #     raise TypeError(f"condition type {self.condition} not supported")
        masked_motion_seq,seq_mask=mask_motion_seq(row_seq,pre_length)
        Motion_embading = self.emb_proj(masked_motion_seq)
        if self.abl_plus:
            emb_latent = Motion_embading + time_emb
        else:
            emb_latent = torch.cat((time_emb, Motion_embading), 0)

        # 4. transformer
        if self.arch == "trans_enc":
            if 0:
                sample = self.pose_embd(sample) #加位置编码，
                xseq = torch.cat((emb_latent, sample), axis=0) #条件嵌入
            else:
                xseq = torch.cat((sample, emb_latent), axis=0)

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            xseq = self.query_pos(xseq) #可学性位置编码
            tokens = self.encoder(xseq)

            if 0:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)

                # zero for padded area
                sample[~mask.T] = 0
            else:
                sample = tokens[:sample.shape[0]]

        elif self.arch == "trans_dec":
            if 0:
                sample = self.pose_embd(sample)

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            if 0:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        return sample


class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions

        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))
        self.guidance_scale = guidance_scale
        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    # cfg核心模块，随机丢弃条件
    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
class EmbedMotion(nn.Module):

    def __init__(self,inputdim=48,
                 seq_len=125,
                 latent_dim=256,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.guidance_scale = guidance_scale
        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask

        self.latent_dim = latent_dim
        self.motion_embedding = nn.Linear(inputdim*seq_len,latent_dim) #暂定，考虑更改探索使用其他结构实现
        self._reset_parameters()
    #应该对动作进行编码到高维操作
    def forward(self, input):

        input=input.reshape(input.shape[0],-1) # [bs, nframes, dim] -> [bs, dim]
        bs = input.shape[0]
        output = self.motion_embedding(input)
        return output.unsqueeze(0)

    # cfg核心模块，随机丢弃条件
    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
#顺序掩码
def mask_motion_seq(motion_seq,masklength: int=100):
    # motion_seq:Tensor [bs, nframes, d]
    #掩码为1，非掩码为0 s
    batch_size,length,dim = motion_seq.shape
    if length < masklength:
        raise ValueError(f"mask length{masklength} should be smaller than sequence length{length}")
    else:
        mask=torch.zeros(batch_size,length).to(motion_seq.device)
        mask[:,length-masklength:]=1
        masked_motion_seq=motion_seq.clone()
        for b in range(batch_size):
            for t in range(length):
                if  mask[b, t]:
                    masked_motion_seq[b, t] = masked_motion_seq[b, t-1]


    return masked_motion_seq,mask
#随机掩码
def mask_motion_rand(motion_seq, mask_ratio=0.1):
    """
    随机mask motion_seq，并用最近的未mask帧填充,优先前向搜索
    motion_seq: [bs, nframes, d]
    mask_ratio: 掩码比例
    return: masked_motion_seq, mask
    """
    bs, nframes, d = motion_seq.shape
    device = motion_seq.device

    # 生成随机mask
    mask = torch.bernoulli(
        torch.ones(bs, nframes, device=device) * mask_ratio
    ).bool()  # [bs, nframes]

    masked_motion_seq = motion_seq.clone()

    for b in range(bs):
        last_valid = None
        for t in range(nframes):
            if mask[b, t]:
                # 如果当前帧被mask
                if last_valid is not None:
                    masked_motion_seq[b, t] = masked_motion_seq[b, last_valid]
                else:
                    # 如果一开始就mask了，找后面第一个未mask的来填
                    future_valid = (mask[b, t:] == 0).nonzero(as_tuple=False)
                    if len(future_valid) > 0:
                        masked_motion_seq[b, t] = motion_seq[b, t + future_valid[0].item()]
                    else:
                        # 全是mask的极端情况，直接置零
                        masked_motion_seq[b, t] = 0
            else:
                last_valid = t

    return masked_motion_seq, mask


if __name__ == '__main__':
    #测试denoiser
    abalation = type('Ablation', (object,), {'MLP_DIST': True, 'PE_TYPE': 'mld','SKIP_CONNECT':True,'DIFF_PE_TYPE':'mld','VAE_TYPE':'yes'})()
    denoi=MldDenoiser(abalation)
    x=torch.randn(64,1,256)
    t=torch.tensor([10])
    c=torch.randn(64,125,48)
    y=denoi(x,t,c)
    print(y[0].shape)

    #测试掩码功能：

    # torch.manual_seed(random.random())  # 可固定随机数，便于复现
    # motion_seq = torch.arange(1, 21, dtype=torch.float32).reshape(1, 10, 2)  # shape [1,8,2]
    #
    # masked_motion_seq, mask = mask_motion_seq(motion_seq, 5)
    #
    # print("原始 motion_seq:\n", motion_seq)
    # print("mask 掩码(1表示mask):\n", mask.int())
    # print("处理后的 masked_motion_seq:\n", masked_motion_seq)

