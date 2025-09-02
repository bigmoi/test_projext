import os
import time
from pathlib import Path
import torch
from diffusers import DDIMScheduler
from utils import GetParament, TrainDiffusion ,logger
from utils.GetParament import GetParam
from models import VAE, Diffusion
import argparse
from utils.datautils import dataset_h36m, dataset_humaneva
import pickle
from torch.utils.tensorboard import SummaryWriter

relative_path = Path.cwd()
print(relative_path.is_dir())





def main(cfg):
    args = GetParam(cfg.cfg) # 读取配置文件.
    loger = logger.create_logger(os.path.join(args.root,'log',f'{time.time()}'+'log.txt'))
    tb_logger = SummaryWriter(os.path.join(args.root,'log','tb_log'))
    abalation = type('Ablation', (object,),
                     {'MLP_DIST': True, 'PE_TYPE': 'mld', 'SKIP_CONNECT': True, 'DIFF_PE_TYPE': 'mld',
                      'VAE_TYPE': 'yes'})()  # 测试使用，之后修改删除分支.
    #dataset initialization
    if args.dataset=='h36m':
        dataset_train = dataset_h36m.DatasetH36M('train', args.t_his, args.t_pred, actions='all')
        dataset_train.getsequancedata()  # 初始化序列数据
        dataloader= torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True,drop_last=True)

        if args.normalize_data:
            dataset_train.normalize_data()

    elif args.dataset=='humaneva':
        1
    else:
        raise ValueError('dataset must be h36m or humaneva')

    #vae model initialization
    vae_model=VAE.Vae(abalation,
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

    if cfg.stage == 'diffusion':
        # 加载预训练vae模型：
        # cp_path = os.path.join(args.root,args.vae_model_path, 'epcoh{}'.format(cfg.iter))
        cp_path=r'E:\MyProjects\test_projext\checkpoints\vae_model\vae_configtest\epcoh99'
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open((cp_path+'.pth'), "rb"))  #暂时使用，后续考虑更改为使用torch.load
        vae_model.load_state_dict(model_cp['model_dict'])

        scheduler = DDIMScheduler(
            num_train_timesteps=args.scheduler.get('params').get('num_train_timesteps'),
                beta_start=args.scheduler.get('params').get('beta_start'),
                beta_end=args.scheduler.get('params').get('beta_end'),
                beta_schedule=args.scheduler.get('params').get('beta_schedule'),
                clip_sample=args.scheduler.get('params').get('clip_sample'),
                set_alpha_to_one=args.scheduler.get('params').get('set_alpha_to_one'),
                steps_offset=args.scheduler.get('params').get('steps_offset')
            )
        scheduler.set_timesteps(args.scheduler.get('num_inference_timesteps'))

        denoiser_model=Diffusion.MldDenoiser(abalation)

        trainer=TrainDiffusion.DiffusionTrainer(
            vae=vae_model,
            model=denoiser_model,
            schedule=scheduler,
            dataloader=dataloader,
            cfg=args,
            logger=loger,
            tb_logger=tb_logger
        )

        trainer.before_train()
        for i in range(0,args.num_denoiser_epoch):

            trainer.before_train_step()
            trainer.run_train_step()
            trainer.after_train_step()

            if  i!=0  and i % args.save_model_interval == 0:
                save_path = os.path.join(args.diffusion_model_path, 'epcoh{}'.format(i))
                os.makedirs(args.diffusion_model_path, exist_ok=True)
                torch.save({'model_dict': denoiser_model.state_dict(),
                            'optimizer_dict': trainer.optimizer.state_dict(),
                            'scheduler_dict': trainer.lr_scheduler.state_dict(),
                            'epoch': i},
                           save_path)
                print('model saved to %s' % save_path)







if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--cfg', default='configtest')
    argparse.add_argument('--stage', default='diffusion')
    argparse.add_argument('--iter', type=int, default=99) #must be  multiple of 100
    cfg = argparse.parse_args()

    main(cfg)
