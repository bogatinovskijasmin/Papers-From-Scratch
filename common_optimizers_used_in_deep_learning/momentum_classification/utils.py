import time
import torch
import torchvision

class FashionMNIST(torch.utils.data.Dataset):
    def __init__(self, batch_size, size=(28, 28)):
        super().__init__()
        self.batch_size = batch_size
        self.root = "../data/"
        transf = torchvision.transforms.Compose([torchvision.transforms.Resize(size=size),
                                                torchvision.transforms.ToTensor()])

        self.train = torchvision.datasets.FashionMNIST(root=self.root,
                                                       train=True,
                                                       transform=transf,
                                                       download=True)

        self.val = torchvision.datasets.FashionMNIST(root=self.root,
                                                     train=False,
                                                     transform=transf,
                                                     download=True)
    def get_dataloader(self, train):
        dataset = self.train if train else self.val
        return torch.utils.data.DataLoader(dataset,
                                           shuffle=train,
                                           batch_size=self.batch_size)
    def get_traindataloader(self):
        return self.get_dataloader(train=True)
    def get_valdataloader(self):
        return self.get_dataloader(train=False)


class Momentum_Scratch(torch.optim.SGD):
    def __init__(self, parameters, lr, momentum):
        super().__init__(parameters, lr)
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.init_states()
    def step(self):
        for param, state in zip(self.parameters, self.states):
            state[:] = self.momentum*state + param.grad
            param.data.sub_(self.lr*state)
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    def init_states(self):
        self.states = []
        for parameter in self.parameters:
            self.states.append(torch.zeros_like(parameter.data))

class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.times.append(time.time()-self.start_time)