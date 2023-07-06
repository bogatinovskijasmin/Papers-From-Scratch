import torch
import numpy as np
import matplotlib.pyplot as plt

from ad_recommendation_data import CTRDataset

def init_layer(module):
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(module.weight)
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)

class FactorizationMachines(torch.nn.Module):
    def __init__(self, field_dims, num_factors):
        super().__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = torch.nn.Embedding(num_inputs, num_factors)
        self.fc = torch.nn.Embedding(num_inputs, 1)
        self.linear_layer = torch.nn.LazyLinear(1, bias=True)

    def forward(self, x):
        # print(f"current {self.embedding(x).shape}")
        square_of_sum = torch.sum(self.embedding(x), dim=1)**2
        sum_of_square = torch.sum(self.embedding(x)**2, dim=1)
        x = self.linear_layer(self.fc(x).sum(1)) + 0.5*(square_of_sum - sum_of_square).sum(1, keepdim=True)
        x = torch.sigmoid(x)
        return x

def get_accuracy(pred, y_true):
    pred = torch.reshape(pred,  (-1, pred.shape[-1]))
    y_true = torch.reshape(y_true, (-1, 1))
    compared = 1.*(pred==y_true)
    return compared.mean()

def train(net, train_iter, test_iter, devices, loss, optimizer, num_epochs):
    loss_ = {}
    for epoch in range(num_epochs):
        print(f"Currently running epoch {epoch+1}")
        train_loss = []
        train_acc = []
        net.train()

        for batch in train_iter:
            optimizer.zero_grad()
            pred = net(batch[0].to(devices))
            pred = pred.reshape(-1)
            l = loss(pred, batch[-1][0].to(devices))
            l.backward()
            optimizer.step()
            acc = get_accuracy(pred, batch[-1][0].to(devices))
            train_loss.append(l.detach().cpu().numpy())
            train_acc.append(acc.detach().cpu().numpy())

        loss_[f"tr_{epoch}"] = np.mean(train_loss)
        loss_[f"tr_acc_{epoch}"] = np.mean(train_acc)
        val_loss = []
        val_acc = []
        net.eval()
        for batch in test_iter:
            pred = net(batch[0].to(devices))
            pred = pred.reshape(-1)
            l = loss(pred, batch[-1][0].to(devices))
            val_loss.append(l.detach().cpu().numpy())
            acc = get_accuracy(pred, batch[-1][0].to(devices))
            val_acc.append(acc.detach().cpu().numpy())

        loss_[f"ts_{epoch}"] = np.mean(val_loss)
        loss_[f"val_acc_{epoch}"] = np.mean(val_acc)
    return loss_

if __name__ == "__main__":
    batch_size = 2048
    num_workers = 4
    data_dir = "../data/ctr/"
    train_data = CTRDataset(data_dir+"train.csv")
    test_data = CTRDataset(data_dir + "test.csv", feat_mapper=train_data.feat_mapper, defaults=train_data.defaults)

    train_iter = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    devices = "cuda:0"
    field_dim = num_featuers = train_data.field_dims
    num_factors = 20
    net = FactorizationMachines(field_dim, num_factors=num_factors)
    net = net.to(devices)
    lr = 0.02
    num_epochs = 30

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.BCEWithLogitsLoss()

    res = train(net, train_iter, test_iter, devices, loss, optimizer, num_epochs)

    plt.plot([res[f"tr_{idx}"] for idx in range(num_epochs)], label="training loss")
    plt.plot([res[f"tr_acc_{idx}"] for idx in range(num_epochs)], label="training acc")

    plt.plot([res[f"ts_{idx}"] for idx in range(num_epochs)], label="test loss")
    plt.plot([res[f"val_acc_{idx}"] for idx in range(num_epochs)], label="test acc")
    plt.title("factorization machines")
    plt.legend()