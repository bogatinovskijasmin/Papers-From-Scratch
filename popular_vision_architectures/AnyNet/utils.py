import torch
import torchvision
from d2l import torch as d2l

class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size, resize=(28, 28)):
        self.resize=resize
        self.batch_size = 64
        self.root = "./"
        self.num_workers = 12
        self.batch_size=batch_size

        transf = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=transf, download=True
        )
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=transf, download=True
        )
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch_size,
                                           shuffle=train,
                                           num_workers=self.num_workers)
    def get_traindataloader(self):
        return self.get_dataloader(train=True)
    def get_valdataloader(self):
        return self.get_dataloader(train=False)
    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def visualize(self, batch, nrows=4, ncols=10, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
