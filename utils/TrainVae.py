import pickle
import sys
import time
import logging
from torch.optim import lr_scheduler
import  os
from models.VAE import Vae
from models.Loss.kl import KLLoss
import torch
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np
from utils.GetParament import  GetParam
from tqdm import tqdm
from datautils.dataset_h36m import DatasetH36M
from datautils.dataset_humaneva import DatasetHumanEva

def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    print('logger created: %s' % filename)
    return logger


def loss_function(input,target,predis,gtdis=torch.distributions.Normal(0,1)):
    KL = KLLoss()
    mse=MSELoss(reduction='mean')

    # klval=  -0.5 * torch.sum(1 + predis.stddev - predis.mean.pow(2) - predis.stddev.exp(), dim=1)
    klval=KL(predis,gtdis)
    mseval=mse(input, target)
    # print("kl:{},mse:{}".format(klval,mseval))
    return mseval-cfg.beta*klval




def TrainVaeOneEpoch(epoch):
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    losses = []
    count=0
    for i in tqdm(generator):
        count+=1
        i=torch.from_numpy(i).float().to(device)
        i=i.reshape(i.shape[0],i.shape[1],-1)
        feats_rst, z, dist=model(i,[cfg.t_his+cfg.t_pred] * cfg.batch_size)
        loss=loss_function(feats_rst,i,dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print('epoch %d, loss mean: %.4f' % (epoch,sum(losses)/len(losses)))
    print(f'一个epoch有{count}个数据')





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configtest')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--iter', type=int, default=0)
    args = parser.parse_args()


    cfg = GetParam(args.cfg)
    '''setup'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    # logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    # print("logger", logger) if logger is not None else print("logger None")
    """parameter"""

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', 25, 100, actions='all', use_vel=False)
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    ablation=dict()
    ablation['MLP_DIST'] = cfg.ablation.get('MLP_DIST', False)
    ablation['PE_TYPE'] = cfg.ablation.get('PE_TYPE', 'mld')
    model = Vae( ablation,
                 nfeatst=48,
                 latent_dim = [1, 256],
                 ff_size = 1024,
                 num_layers = 9,
                 num_heads = 4,
                 dropout = 0.1,
                 arch = "encoder_decoder",
                 normalize_before = False,
                 activation = "gelu",
                 position_embedding = "learned")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.vae_lr)


    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - cfg.num_vae_epoch_fix) / float(cfg.num_vae_epoch - cfg.num_vae_epoch_fix + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


    # 读取检查点
    if args.iter > 0:
        cp_path = os.path.join(cfg.vae_model_path,'epcoh{}'.format(args.iter))
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    testckpt = False
    if testckpt :
        test_ckpt_path=r'E:\MyProjects\test_projext\checkpoints\1222_mld_humanml3d_FID041.ckpt'
        model_cp = torch.load(open(test_ckpt_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if cfg.mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            TrainVaeOneEpoch(i)
            # 保存检查点处
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                os.makedirs(cfg.vae_model_path, exist_ok=True)
                cp_path =os.path.join(cfg.vae_model_path,'epcoh{}.pth'.format(i))
                model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                # pickle.dump(model_cp, open(cp_path, 'wb'))
                torch.save(model_cp, cp_path)
                # logger.info('save model to %s' % cp_path)