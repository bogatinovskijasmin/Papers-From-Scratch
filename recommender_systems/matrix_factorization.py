import collections

import numpy as np
import torch

import matplotlib.pyplot as plt
from utils import split_and_load_ml100k

def init_network(module):
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, 0, 0.01)
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        self.Q = torch.nn.Embedding(num_users, num_factors)
        self.P = torch.nn.Embedding(num_items, num_factors)
        self.b_u = torch.nn.Embedding(num_users, 1)
        self.b_i = torch.nn.Embedding(num_items, 1)

        self.Q.apply(init_network)
        self.P.apply(init_network)
        self.b_u.apply(init_network)
        self.b_i.apply(init_network)
    def forward(self, user_id, item_id):
        P = self.P(item_id)
        Q = self.Q(user_id)
        b_u = self.b_u(user_id)
        b_i = self.b_i(item_id)
        output = (P*Q).sum(dim=1, keepdims=True) + b_u + b_i
        return output.flatten()

def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs, devices="cuda:0", **kwargs):
    loss_ = collections.defaultdict(dict)
    for epoch in range(num_epochs):
        d = []
        net.train()
        print(f"Currently running epoch {epoch+1}")
        for i, values in enumerate(train_iter):
            trainer.zero_grad()
            preds = net(values[0], values[1])
            true_vals = values[-1].type(torch.float32)
            l = torch.sqrt(loss(preds, true_vals).mean())
            l.backward()
            trainer.step()
            d.append(l.item())
        loss_["tr_"+str(epoch)]= np.mean(d)
        d = []
        net.eval()
        with torch.no_grad():
            for i, values in enumerate(test_iter):
                preds = net(values[0], values[1])
                true_vals = values[-1].type(torch.float32)
                l = torch.sqrt(loss(preds, true_vals).mean())
                d.append(l.item())
            loss_["ts_"+str(epoch)]= np.mean(d)
    return loss_

if __name__ == "__main__":

    num_users, num_items, train_iter, test_iter = split_and_load_ml100k(test_ratio=0.1, batch_size=512)
    devices = "cuda:0"
    net = MatrixFactorization(num_factors=30, num_users=num_users, num_items=num_items)

    lr, num_epochs, wd = 0.002, 100, 1e-5
    loss = torch.nn.MSELoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    res = train_recsys_rating(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, trainer=trainer, num_epochs=num_epochs, devices="cuda")

    plt.plot([res["tr_" + str(epoch)] for epoch in range(num_epochs)], label="training loss")
    plt.plot([res["ts_"+str(epoch)] for epoch in range(num_epochs)], label="eval loss")
    plt.grid()
    plt.legend()

    user_id = 20
    item_id = 30

    print(f"The user with id {user_id} will give recommendation of {int(net(torch.tensor([20], dtype=torch.int), torch.tensor([30], dtype=torch.int)).item())} for the item with id {item_id}.")
