import math

import torch

class SGD_Scratch(torch.optim.SGD):
    def __init__(self, parameters, lr=0.01):
        self.lr = lr
        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.grad.zero_()
    def step(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param[:] -= self.lr*param.grad



class Momentum_Scratch(torch.optim.SGD):
    def __init__(self, parameters, momentum=0.5, lr=0.01):
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        self.init_states()
    def init_states(self):
        self.states = []
        for param in self.parameters():
            if param.requires_grad == True:
                self.states.append(torch.zeros_like(param))
    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.grad.zero_()
    def step(self):
        for param, state in zip(self.parameters(), self.states):
            state[:] = self.momentum*state + (1-self.momentum)*param.grad
            param[:] -= self.lr*state


class AdaGrad_Scratch(torch.optim.SGD):
    def __init__(self, parameters, beta=0.8, momentum=0.5, lr=0.01):
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        self.beta = 1
        self.eta = 10**(-5)
        self.init_states()

    def init_states(self):
        self.states = []
        for param in self.parameters():
            if param.requires_grad == True:
                self.states.append(torch.zeros_like(param))

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.grad.zero_()

    def step(self):
        for param, state in zip(self.parameters(), self.states):
            state[:] = state + param.grad**2
            param[:] -= (self.lr/(torch.sqrt(state) + self.eta))*param.grad
class RMSProp_Scratch(torch.optim.SGD):
    def __init__(self, parameters, beta=0.8, momentum=0.5, lr=0.01):
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        self.beta = beta
        self.eta = 10 ** (-5)
        self.init_states()
    def init_states(self):
        self.states = []
        for param in self.parameters():
            if param.requires_grad == True:
                self.states.append(torch.zeros_like(param))
    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.grad.zero_()
    def step(self):
        for param, state in zip(self.parameters(), self.states):
            state[:] = self.beta*state + (1-self.beta)*param.grad**2
            param[:] -= (self.lr/(torch.sqrt(state)+ self.eta))*param.grad



class ADAM_Scratch(torch.optim.SGD):
    def __init__(self, parameters, beta=0.8, momentum=0.5, lr=0.01, beta2=0.99):
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        self.beta = beta
        self.beta2 = beta2
        self.eta = 10 ** (-5)
        self.init_states()
    def init_states(self):
        self.states = []
        self.velocity = []
        for param in self.parameters():
            if param.requires_grad == True:
                self.states.append(torch.zeros_like(param))
                self.velocity.append(torch.zeros_like(param))

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad == True:
                param.grad.zero_()

    def step(self, t):
        for param, state, velo in zip(self.parameters(), self.states, self.velocity):
            velo[:] = self.beta * velo + (1 - self.beta) * param.grad
            state[:] = self.beta2*state + (1-self.beta2)*param.grad**2
            velo_rescaled = velo/(1-math.pow(self.beta, t))
            state_rescaled = state/(1-math.pow(self.beta2, t))
            param[:] -= self.lr*velo_rescaled/(torch.sqrt(state_rescaled)+self.eta)


