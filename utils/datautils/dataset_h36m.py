import numpy as np
import os

from cv2 import batchDistance

from utils.datautils.dataset import Dataset
from utils.datautils.skeleton import Skeleton
from pathlib import Path
class DatasetH36M(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False, batch_siz=8,removed_joints=None):
        self.use_vel = use_vel
        self.removed_joints = removed_joints
        super().__init__(mode, t_his, t_pred, actions,batch_siz)
        if use_vel:
            self.traj_dim += 3


    def prepare_data(self):
        self.data_file = os.path.join(Path(__file__).parent.parent.parent,'data/data', 'data_3d_h36m.npz')
        print(f"当前目录为%s"%os.getcwd())
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [9, 11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        if self.removed_joints==None:
            self.removed_joints = {0,4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}

        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self.process_data()
    # get data
    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()

        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f



def compute_velocity(joints):
    """
    计算关节速度
    joints: (T, J, 3) numpy 数组
    返回: (T-1, J, 3) 每一帧相对前一帧的速度向量
    """
    velocity = joints[1:] - joints[:-1]
    return velocity


def compute_joint_angle(joints, parent, joint, child):
    """
    计算关节角度 (三点夹角)
    joints: (T, J, 3)
    parent, joint, child: int, 三个关节的索引
    返回: (T,) 每一帧对应的夹角 (弧度制)
    """
    vec1 = joints[:, parent] - joints[:, joint]  # parent -> joint
    vec2 = joints[:, child] - joints[:, joint]  # child -> joint

    # 归一化
    vec1_norm = vec1 / np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=-1, keepdims=True)

    # 点积求夹角
    dot = np.sum(vec1_norm * vec2_norm, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)  # 避免数值误差超出 [-1,1]

    angle = np.arccos(dot)  # 弧度制
    return angle
if __name__ == '__main__':
    # np.random.seed(0)
    # actions = 'all'
    # dataset = DatasetH36M('train', actions=actions,batch_siz=16)
    # generator = dataset.sampling_generator(num_samples=1000, batch_size=16)
    # dataset.normalize_data()
    # # generator = dataset.iter_generator()
    # i=0
    # for data in generator:
    #     i+=1
    #     print(data.shape)
    # print(i)

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    # 假设 DatasetH36M 已按要求实现
    dataset = DatasetH36M('train', actions='all')
    dataset.prepare_data()  # 确保数据已加载
    dataset.getsequancetata()#初始化顺序数据

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,drop_last=True)

    for batch in tqdm(dataloader):
        batch=batch.reshape(batch.shape[0], batch.shape[1], -1)
        print(batch.shape)  # 检查每个 batch 的形状
        #   # 只取第一个 batch 测试
        # for i in range(batch.shape[0]):
        #     joints = batch[i]
        #     speed=compute_velocity(joints.numpy())
        #     angles=compute_joint_angle(joints.numpy(), parent=8, joint=11, child=14)
        #     print(f"Sample {i} - Speed shape: {speed}, Angles shape: {angles}")

