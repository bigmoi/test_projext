
# def getmodel(config,dataset):
#     if config.model_name=='VAE':
#         from .VAE import Vae
#         traj_dim = dataset.traj_dim
#         model = Vae(config,traj_dim,load_path=config.pretrained_vae_path)
#     else:
#         raise NotImplementedError
#     return model