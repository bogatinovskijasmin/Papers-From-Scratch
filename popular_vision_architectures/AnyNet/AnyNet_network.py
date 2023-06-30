import torch

from main_class import Classifier
from residual_network import ResNeXtBlock

class AnyNet(Classifier):
    def stem(self, num_channels):
        return torch.nn.Sequential(
            torch.nn.LazyConv2d(out_channels=num_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU()
        )

    def stage(self, depth, num_channels, groups, bot_mul):
        blk = []
        for i in range(depth):
            if i==0:
                blk.append(ResNeXtBlock(num_channels=num_channels, groups=groups, bot_mul=bot_mul, use_1x1conv=True, strides=2))
            else:
                blk.append(ResNeXtBlock(num_channels=num_channels, groups=groups, bot_mul=bot_mul))
        return torch.nn.Sequential(*blk)


    def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
        super().__init__(lr=lr)
        self.net = torch.nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module("head", torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(num_classes)
        ))

    def forward(self, X):
        return self.net(X)
    def apply_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)


class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
