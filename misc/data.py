import time
import numpy as np
import torch
from d2l import torch as d2l

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat', '76e5be1548fd8222e5074cf0faae75edff8cf93f')
class Data:
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_iter, self.feature_len = self.get_data_ch11(batch_size=batch_size)
        self.X, self.y = self.__create_data()
    def get_data_ch11(self, batch_size, n=1500):
        data = np.genfromtxt(d2l.download('airfoil'), dtype=np.float32, delimiter='\t')
        data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
        data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
        return data_iter, data.shape[1] - 1
    def __create_data(self, ):
        tmp1 = []
        tmp2 = []
        for X, y in self.data_iter:
            tmp1.append(X)
            tmp2.append(y)
        return torch.vstack(tmp1), torch.hstack(tmp2)
    def get_tensorloader(self, tensors):
        tensors = tuple(tensor for tensor in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
    def get_dataloader(self):
        return self.get_tensorloader((self.X, self.y))
    def get_traindataloader(self):
        return self.get_dataloader()
    def get_valdataloader(self):
        return self.get_dataloader()


class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.times.append(time.time()-self.start_time)