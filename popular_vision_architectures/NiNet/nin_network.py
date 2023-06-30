import torch
from main_class import Classifier
def nin_block(out_channels, kernel_size, strides, padding):
    return torch.nn.Sequential(
        torch.nn.LazyConv2d(out_channels, kernel_size, strides, padding),
        torch.nn.ReLU(),
        torch.nn.LazyConv2d(out_channels, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.LazyConv2d(out_channels, kernel_size=1),
        torch.nn.ReLU(),
    )
def init_method(module):
    if type(module) == torch.nn.Conv2d or type(module) == torch.nn.LazyLinear:
        torch.nn.init.xavier_uniform_(module.weight)
class NiN(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(lr=lr)

        self.num_classes = num_classes
        self.net = torch.nn.Sequential(nin_block(96, kernel_size=11, strides=4, padding=0),
                                       torch.nn.MaxPool2d(3, stride=2),
                                       nin_block(256, kernel_size=5, strides=1, padding=2),
                                       torch.nn.MaxPool2d(3, stride=2),
                                       nin_block(384, kernel_size=3, strides=1, padding=1),
                                       torch.nn.MaxPool2d(3, stride=2),
                                       torch.nn.Dropout(0.5),
                                       nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                                       torch.nn.AdaptiveAvgPool2d((1, 1)),
                                       torch.nn.Flatten()
                                       )
    def forward(self, X):
        return self.net(X)

    def network_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)