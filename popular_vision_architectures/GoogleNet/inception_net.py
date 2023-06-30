import torch
from main_class import Classifier

def init_cnn(module):
    if type(module) is torch.nn.Linear or type(module) is torch.nn.LazyConv2d:
        torch.nn.init.xavier_uniform_(module.weight)

class Inception(torch.nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        self.b1_1 = torch.nn.LazyConv2d(c1, kernel_size=1)
        self.b2_1 = torch.nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = torch.nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        self.b3_1 = torch.nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = torch.nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        self.b4_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = torch.nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = torch.nn.functional.relu(self.b1_1(x))
        b2 = torch.nn.functional.relu(self.b2_2(torch.nn.functional.relu(self.b2_1(x))))
        b3 = torch.nn.functional.relu(self.b3_2(torch.nn.functional.relu(self.b3_1(x))))
        b4 = torch.nn.functional.relu(self.b4_2(self.b4_1(x)))
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogleNet(Classifier):
    def b1(self):
        return torch.nn.Sequential(
            torch.nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b2(self):
        return torch.nn.Sequential(
            torch.nn.LazyConv2d(64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(192, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b3(self):
        return torch.nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                                   Inception(128, (128, 192), (32, 96), 64),
                                   torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                   )

    def b4(self):
        return torch.nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                   )


    def b5(self):
        return torch.nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             torch.nn.AdaptiveAvgPool2d((1,1)),
                                   torch.nn.Flatten())

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(lr=lr)
        self.lr = lr
        self.num_classes = num_classes

        self.net = torch.nn.Sequential(
            self.b1(),
            self.b2(),
            self.b3(),
            self.b4(),
            self.b5(),
            torch.nn.LazyLinear(num_classes)
        )
    def forward(self, x):
        return self.net(x)
    def network_init(self, inputs, init_method=None):
        self.forward(*inputs)
        if init_method is not None:
            self.net.apply(init_method)