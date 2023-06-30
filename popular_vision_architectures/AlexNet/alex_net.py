import torch
from  main_class import Classifier


def init_cnn(module):
    if type(module) is torch.nn.Linear or type(module) is torch.nn.LazyConv2d:
        torch.nn.init.xavier_uniform_(module.weight)
class AlexNet(Classifier):
    def __init__(self, num_classes, lr):
        super().__init__(lr=lr)
        self.lr = lr

        self.net = torch.nn.Sequential(
            torch.nn.LazyConv2d(out_channels=96, kernel_size=11, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.LazyLinear(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.LazyLinear(num_classes)
        )
    def forward(self, X):
        return self.net(X)

    def apply_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)

