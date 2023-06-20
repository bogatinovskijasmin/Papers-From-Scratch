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


class SGD_Scratch(torch.optim.SGD):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for param in self.parameters:
            param.data.sub_(self.lr*param.grad)
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

class RMSProp_Scratch(torch.optim.Adadelta):
    def __init__(self, parameters, lr, beta):
        super().__init__(parameters, lr, beta)
        self.parameters = parameters
        self.lr = lr
        self.beta = beta
        self.eta = 10**(-15)
        self.states = []
        self.init_state()
    def init_state(self):
        for param in self.parameters:
            self.states.append(torch.zeros_like(param.data))
    def step(self):
        for param, state in zip(self.parameters, self.states):
            state[:] = self.beta*state + (1-self.beta)*torch.square(param.grad)
            param[:] = param.data - self.lr/(torch.sqrt(state)+self.eta)*param.grad
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.times.append(time.time()-self.start_time)