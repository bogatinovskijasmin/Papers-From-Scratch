import torch
from main_class import Classifier
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(torch.nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return torch.nn.Sequential(*layers)

def init_method(module):
    if type(module) == torch.nn.Conv2d or type(module) == torch.nn.LazyLinear:
        torch.nn.init.xavier_uniform_(module.weight)
class VGG(Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__(lr=lr)
        self.arch = arch
        self.num_classes = num_classes
        conv_blks = []

        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = torch.nn.Sequential(*conv_blks,
                                       torch.nn.Flatten(),
                                       torch.nn.LazyLinear(128),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.LazyLinear(128),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.LazyLinear(num_classes)
                                       )
    def forward(self, X):
        return self.net(X)

    def network_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)