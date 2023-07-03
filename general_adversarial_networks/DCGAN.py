import collections
import torch
import torchvision

from d2l import torch as d2l
import numpy as np
from sample_gan import update_D, update_G

class G_block(torch.nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_terans = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.ReLU()
    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_terans(X)))

class D_block(torch.nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device="cuda:0"):
    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    for w in net_D.parameters():
        torch.nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        torch.nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)

    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999))
    losses = collections.defaultdict()

    for epoch in range(num_epochs):
        print(f"Number of epochs {epoch+1}")
        d_D = []
        d_G = []
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            loss_D = update_D(X, Z, net_D, net_G, loss, trainer_D)
            loss_G = update_G(Z, net_D, net_G, loss, trainer_G)
            d_D.append(loss_D.detach().cpu().numpy()/batch_size)
            d_G.append(loss_G.detach().cpu().numpy()/batch_size)
        losses[f"Epoch{str(epoch)}_loss_D"] = np.mean(d_D)
        losses[f"Epoch{str(epoch)}_loss_G"] = np.mean(d_G)
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        fake_X = net_G(Z).permute(0, 2, 3, 1).detach().cpu().numpy()/2+0.5
        losses[f"Epoch{str(epoch)}_samples"] = fake_X

    return losses

if __name__ == "__main__":
    ## Dataset loading
    batch_size = 256

    d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip', 'c065c0e2593b8b161a2d7873e42418bf6a21106c')
    data_dir = d2l.download_extract('pokemon')
    pokemon = torchvision.datasets.ImageFolder(data_dir)

    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)])
    pokemon.transform = transformer
    data_iter = torch.utils.data.DataLoader(pokemon, batch_size=batch_size, shuffle=True, num_workers=4)


    ## Model Initialization
    n_G = 64
    n_D = 64
    latent_dim = 100
    lr = 0.005
    num_epochs = 20

    net_G = torch.nn.Sequential(
        G_block(in_channels=100, out_channels=n_G*8, strides=1, padding=0),
        G_block(in_channels=n_G*8, out_channels=n_G*4),
        G_block(in_channels=n_G*4, out_channels=n_G*2),
        G_block(in_channels=n_G*2, out_channels=n_G),
        torch.nn.ConvTranspose2d(in_channels=n_G, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
        torch.nn.Tanh())

    net_D = torch.nn.Sequential(
        D_block(n_D),
        D_block(in_channels=n_D, out_channels=n_D*2),
        D_block(in_channels=n_D*2, out_channels=n_D*4),
        D_block(in_channels=n_D*4, out_channels=n_D*8),
        torch.nn.Conv2d(in_channels=n_D*8, out_channels=1, kernel_size=4, bias=False))

    results = train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
