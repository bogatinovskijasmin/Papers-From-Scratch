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


class ADAdelta_Scratch(torch.optim.Adadelta):
    def __init__(self, parameters, rho):
        super().__init__(parameters, rho)
        self.parameters = parameters
        self.rho = rho
        self.states = []
        self.eta = 10**(-6)
        self.intermidiate_states = []

        self.init_states()
    def step(self):
        for param, state, intermidiate_state in zip(self.parameters, self.states, self.intermidiate_states):
            state[:] = self.rho*state + (1-self.rho)*torch.square(param.grad)
            deltas = param.grad*torch.sqrt(intermidiate_state+ self.eta)
            squared_deltas = torch.sqrt(state+ self.eta)
            update_grad = deltas.div_(squared_deltas)
            param[:] -= update_grad
            intermidiate_state[:] = self.rho*intermidiate_state - (1-self.rho)*torch.square(update_grad)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def init_states(self):
        for param in self.parameters:
            self.states.append(torch.zeros_like(param))
            self.intermidiate_states.append(torch.zeros_like(param))

class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.times.append(time.time()-self.start_time)