import collections
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(data, batch_size, is_train):
    data = torch.utils.data.TensorDataset(*data)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=is_train)

def train(net_G, net_D, data_iter, lr_G, lr_D, lattent_dim, num_epochs):
    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
    for w in net_D.parameters():
        torch.nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        torch.nn.init.normal_(w, 0, 0.02)

    net_G_optim = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    net_D_optim = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    training_loss = collections.defaultdict()

    for epoch in range(num_epochs):
        print(f"currently running {epoch+1}")
        loss_D_list = []
        loss_G_list = []
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, (batch_size, lattent_dim))
            loss_D = update_D(X, Z, net_D, net_G, net_D_optim, net_G_optim, loss)
            loss_G = update_G(Z, net_D, net_G, net_G_optim, loss)
            loss_D_list.append(loss_D.detach().numpy())
            loss_G_list.append(loss_G.detach().numpy())

        training_loss[f"Epoch{epoch}_loss_D"] = np.mean(loss_D_list)
        training_loss[f"Epoch{epoch}_loss_G"] = np.mean(loss_G_list)
        z = torch.normal(0, 1, (100, lattent_dim))
        training_loss[f"Epoch{epoch}_sampled_data"] = net_G(z)
    return training_loss

def update_D(X, Z, net_D, net_G, net_D_optim, net_G_optim, loss):
    batch_size = X.shape[0]
    net_D_optim.zero_grad()

    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)

    true_y = net_D(X)
    fake_x = net_G(Z)
    fake_y = net_D(fake_x)

    loss_D = (loss(true_y, ones.reshape(true_y.shape)) + loss(fake_y, zeros.reshape(true_y.shape)))/2

    loss_D.backward()
    net_D_optim.step()
    return loss_D

def update_G(Z, net_D, net_G, net_G_optim, loss):
    batch_size = Z.shape[0]
    net_G_optim.zero_grad()

    ones = torch.ones((batch_size,), device=Z.device)
    fake_x = net_G(Z)
    fake_y = net_D(fake_x)

    loss_G = loss(fake_y, ones.reshape(fake_y.shape))
    loss_G.backward()
    net_G_optim.step()
    return loss_G



if __name__ == "__main__":



    ### DEFINE DATA
    num_samples = 1000
    num_dimensions = 2
    batch_size = 8

    X = torch.normal(0, 1, (num_samples, num_dimensions))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b
    data_iter = load_data((data,), batch_size=batch_size, is_train=True)


    ### DEFINE MODELS
    lattent_dim = 2
    net_G = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
    )
    net_D = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 1),
    )
    ### TRAIN GAN
    lr_D = 0.1
    lr_G = 0.05
    num_epochs = 20
    results = train(net_G, net_D, data_iter, lr_G = lr_G, lr_D=lr_D, lattent_dim=lattent_dim, num_epochs=num_epochs)


    ### PLOT RESULTS
    plt.close("all")
    plt.scatter(data[:100, 0], data[:100, 1], label="ground truth")
    plt.scatter(results["Epoch19_sampled_data"].detach().numpy()[:, 0], results["Epoch19_sampled_data"].detach().numpy()[:, 1], label="pred")
    plt.legend()
    plt.show()

