import math
import torch
from optimizers import SGD_Scratch, Momentum_Scratch, AdaGrad_Scratch, RMSProp_Scratch, ADAM_Scratch
from torch.optim import SGD, Adagrad, Adadelta, RMSprop, Adam

class Regression(torch.nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.d_input = d_input
    def forward(self):
        raise NotImplementedError
    def loss(self, ):
        raise NotImplementedError
    def configure_optimizer(self):
        raise NotImplementedError
    def training_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        a = self.loss(self.forward(*batch[:-1]), batch[-1])
        return l
    def validation_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        return l
    def checkpoint_model(self):
        raise NotImplementedError


class LinearRegressionScratch(Regression):
    def __init__(self, d_input, lr=0.01, momentum=0.8, beta=0.8, beta2=0.99):
        super().__init__(d_input=d_input)
        self.d_input = d_input
        self.lr = lr
        self.beta = beta
        self.beta2 = beta2
        self.momentum = momentum
        self.W = torch.normal(0, 0.01, (d_input, 1), requires_grad=True, device="cuda:0")
        self.b = torch.zeros((1), requires_grad=True, device="cuda:0")\

    def forward(self, X):
        return torch.matmul(X, self.W) + self.b
    def parameters(self):
        return [self.W, self.b]

    # def configure_optimizer(self):
    #     # return SGD_Scratch(self.parameters, lr=self.lr)
    #     # return Momentum_Scratch(self.parameters, lr=self.lr, momentum=self.momentum)
    #     # return AdaGrad_Scratch(self.parameters, lr=self.lr)
    #     # return RMSProp_Scratch(self.parameters, lr=self.lr, beta=self.beta)
    #     return ADAM_Scratch(self.parameters, lr=self.lr, beta=self.beta, beta2=self.beta2)

    def configure_optimizer(self):
        return SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    #     return Adagrad(self.parameters(), lr=self.lr)
    #     return RMSprop(self.parameters(), lr=self.lr, alpha=self.beta)
        return Adam(self.parameters(), lr=self.lr, betas=(self.beta, self.beta2))



    def loss(self, y_pred, y):
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        return 0.5*torch.mean((y_pred-y)**2)
    def checkpoint_model(self, checkpoint_name="initial_run.pth"):
        torch.save(self.state_dict(), checkpoint_name)
    def get_checkpoint_model(self, checkpoint_name="initial_run.pth"):
        return self.load_state_dict(torch.load(checkpoint_name))