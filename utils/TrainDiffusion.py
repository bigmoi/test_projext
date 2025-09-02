import copy
import os
# import diffusers.DDPMScheduler as DDPMScheduler
import time
import torch
import torch.optim as optim
from diffusers.models.autoencoders import vae
from torch import nn as nn
from tqdm import tqdm
from  einops import rearrange
from utils.logger import AverageMeter
from models.VAE import Vae
import models.Diffusion as Diffusion


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class DiffusionTrainer:
    def __init__(self,
                 vae,
                 model,
                 schedule,
                 dataloader,
                 cfg,
                 logger,
                 tb_logger):
        # super().__init__()
        self.vae = vae # Vae should be pre-trained  and frozen during diffusion training
        self.diffusion_denoiser = model
        self.noise_scheduler = schedule # noise schedule
        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None


        self.dataloader = dataloader
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 1
        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        self.diffusion_denoiser.to(self.cfg.device)

        #冻结vae参数

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.to(self.cfg.device)
    def before_train(self):
        self.optimizer = optim.AdamW(self.diffusion_denoiser.parameters(), lr=self.cfg.diffusionlr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
                                                              gamma=0.8)
        self.noise_pre_loss = nn.MSELoss()
        self.noise_pre_loss_weight = 1.0

        self.data_pre_loss = nn.MSELoss()
        self.data_pre_loss_weight = 1.0
    def before_train_step(self):
        self.diffusion_denoiser.train()

        self.vae.eval()
        # self.generator_train =1 # todo 占位符，后续改为可迭代数据生成器. 使用dataloader
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")
    def run_train_step(self): #实现前向传播过程和反向传播过程
        for sample in tqdm(self.dataloader):
            sample=rearrange(sample,'b t j c -> b t (j c)') # [B, T, joints*c]
            sample = sample.to(self.cfg.device)

            with torch.no_grad():
                latent, dist=self.vae.encode(sample,[self.cfg.t_his+self.cfg.t_pred] * self.cfg.batch_size)

            latent = latent.permute(1, 0, 2) #[n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
            bsz = latent.shape[0]
            # diffusion process
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latent.device,
            )

            noise=torch.randn_like(latent).to(self.cfg.device)
            noisy_latents = self.noise_scheduler.add_noise(latent.clone(), noise,timesteps)
            pred_noise=self.diffusion_denoiser(noisy_latents,timesteps,sample,lengths=None,pre_length=100,)

            loss= self.noise_pre_loss_weight * self.noise_pre_loss(noise, pred_noise) # +self.data_pre_loss_weight * self.data_pre_loss(timesteps, pred_noise))
            self.train_losses.update(loss.item()) # 记录损失值
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]
            if args_ema is True:
                ema.step_ema(ema_model, self.diffusion_denoiser)

            del loss,sample, noisy_latents, pred_noise, noise, latent, dist

    def after_train_step(self): # 更新lr调度，记录日志信息
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))

        if self.iter in [1,2,3,4,5] or self.iter % self.cfg.save_model_interval == 0: #=1 4 test
            self.save_checkpoint(os.path.join(self.cfg.root,self.cfg.diffusion_model_path))
        self.iter = self.iter + 1


    def before_val(self):
        1
    def run_val_step(self):
        1
    def after_val_step(self):
        1

    def save_checkpoint(self,ckpt_path):
        """保存检查点，支持最佳模型保存"""
        # 确保保存目录存在
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # 准备保存内容
        checkpoint = {
            'iter': self.iter,
            'model_state_dict': self.diffusion_denoiser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses.avg,
            'lrs': self.lrs,
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
            'cfg': self.cfg,  # 保存配置便于恢复
        }

        # 保存定期检查点
        checkpoint_path = f'{ckpt_path}/Epoch{self.iter}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'保存检查点到 {ckpt_path}')

        # 如果是最佳模型，额外保存一份
        # if is_best:
        #     best_path = f'{self.cfg.ckpt}_best.pt'
        #     torch.save(checkpoint, best_path)
        #     self.logger.info(f'保存最佳模型到 {best_path}')

    def load_checkpoint(self, ckpt_path):
        """加载检查点恢复训练"""
        if not os.path.exists(ckpt_path):
            self.logger.warning(f"检查点 {ckpt_path} 不存在")
            return False

        self.logger.info(f"从检查点 {ckpt_path} 恢复训练")
        checkpoint = torch.load(ckpt_path, map_location=self.cfg.device)

        self.before_train()
        # 恢复模型参数
        self.diffusion_denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        # 恢复EMA模型（如果有）
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        # 恢复训练状态
        self.iter = checkpoint['iter']
        if 'lrs' in checkpoint:
            self.lrs = checkpoint['lrs']

        self.logger.info(f"成功恢复至Epoch {self.iter}")
        return True

if __name__ == '__main__':
    ##test
    1