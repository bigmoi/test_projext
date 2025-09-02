import numpy as np
import torch
class Dataset(torch.utils.data.Dataset):

    def __init__(self, mode, t_his, t_pred, actions='all',batch_size=8):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.batch_size = batch_size

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def sample(self):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        return traj[None, ...]

    def sampleboth(self):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start: fr_end]
        x=traj[0:self.t_his,:,:]
        y=traj[self.t_his:self.t_pred,:,:]
        return x,y


    def sampling_generator(self, num_samples=1000, batch_size=16):
        # 用途：随机采样训练数据
        # 采样方式：完全随机采样，每次从任意subject和action中随机选择一个时间窗口
        # 数据分布：每个batch中的样本来源不同，数据多样性高
        # batch_size=self.batch_size
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)

            yield sample

    def iter_generator(self, step=25):
        # 用途：顺序遍历所有数据
        # 采样方式：按固定步长顺序遍历每个序列的每个时间窗口
        # 数据分布：按序列顺序返回，保证数据的完整覆盖
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]

                    yield traj


    def getsequancedata(self):
        # 用途：按照batch获取数据集的所有数据
        # 设定step=25，按步长25顺序遍历所有数据
        all_data = list(self.iter_generator(step=25))
        # self.all_data_array_len = sum([data.shape[0] for data in all_data])
        self.all_data_array = np.concatenate(all_data, axis=0)
        self.all_data_array_len =len(self.all_data_array)
        return self.all_data_array_len if hasattr(self, 'self.all_data_array_len') else 0

    def __len__(self):

        return self.all_data_array_len


    def __getitem__(self, idx):
        return self.all_data_array[idx]








