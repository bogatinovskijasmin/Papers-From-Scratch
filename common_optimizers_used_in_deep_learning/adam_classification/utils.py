import time
import torch
import torchvision
from d2l import torch as d2l 

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


class Adam_scratch(torch.optim.Adam):
    def __init__(self, parameters, lr, beta1, beta2):
        super().__init__(parameters, lr)
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.velocity = []
        self.states = []
        self.step_ = 1
        self.eta = 10**(-8)

        self.init_states()

    def init_states(self):
        for param in self.parameters:
            self.velocity.append(torch.zeros_like(param))
            self.states.append(torch.zeros_like(param))
    def step(self):
        for param, velocity, states in zip(self.parameters, self.velocity, self.states):
            velocity[:] = self.beta1*velocity + (1.-self.beta1)*param.grad
            states[:] = self.beta2*states + (1.-self.beta2)*torch.square(param.grad)
            vel_rescaled = velocity.div_(1.-torch.pow(torch.tensor(self.beta1), self.step_))
            state_rescaled = states.div_(1.-torch.pow(torch.tensor(self.beta2), self.step_))
            gradient_scaled_numerator = vel_rescaled.mul_(self.lr)
            gradient_scaled_denumerator = state_rescaled.sqrt_() + self.eta
            gradinet_scaled = gradient_scaled_numerator.div_(gradient_scaled_denumerator)
            param.data.sub_(gradinet_scaled)
        self.step_+=10
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