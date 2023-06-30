import torch
import torch.nn as nn

from main_class import Classifier

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X-mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X-mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X-mean)/torch.sqrt(var+eps)
        moving_mean = (1.0-momentum)*moving_mean + momentum*mean
        moving_var = (1.0-momentum)*moving_var + momentum*var
    Y = gamma*X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self, X):
        if self.moving_mean.device !=X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.1)
        return Y
class BNLeNetScratch(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(lr=lr)
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
    def forward(self, X):
        return self.net(X)
    def apply_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)

class BNLeNet(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(lr=lr)
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), torch.nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), torch.nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
    def forward(self, X):
        return self.net(X)
    def apply_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)
def init_cnn(module):
    if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)