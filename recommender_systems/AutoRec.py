import torch
import  numpy as np
import collections
import matplotlib.pyplot as plt


from utils import read_data_ml100k, load_data_ml100k, split_data_ml100k

def init_layers(module):

    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, 0.0, 0.01)
        torch.nn.init.zeros_(module.bias)

class AutoRec(torch.nn.Module):
    def __init__(self, num_hiddens, num_users, dropout=0.05):
        super().__init__()
        self.encoder = torch.nn.LazyLinear(num_hiddens, bias=True)
        self.activation1 = torch.nn.Sigmoid()
        self.decoder = torch.nn.LazyLinear(num_users, bias=True)
        self.dropout = torch.nn.Dropout(dropout)

        X = torch.ones((1, num_users), dtype=torch.float32)
        for module in self.modules():
            X = module(X)
        self.encoder.apply(init_layers)
        self.decoder.apply(init_layers)
    def forward(self, ins):
        hidden = self.activation1(self.dropout(self.encoder(ins)))
        pred = self.decoder(hidden)
        if self.training:
            return pred*torch.sign(ins)
        else:
            return pred

def train_autorec(net, train_iter, test_iter, loss, trainer, num_epochs, devices="cuda:0", **kwargs):
    loss_ = collections.defaultdict(dict)

    for epoch in range(num_epochs):
        d = []
        net.train()
        print(f"Currently running epoch {epoch+1}")
        for i, values in enumerate(train_iter):
            trainer.zero_grad()
            preds = net(values)
            true_vals = values.type(torch.float32)
            l = torch.sqrt(torch.sum(loss(torch.sign(true_vals)*preds, true_vals))/torch.sign(true_vals).sum())
            l.backward()
            trainer.step()
            d.append(l.item())
        loss_["tr_"+str(epoch)]= np.mean(d)
        d = []
        net.eval()
        with torch.no_grad():
            for i, values in enumerate(test_iter):
                preds = net(values)
                true_vals = values.type(torch.float32)
                l = torch.sqrt(torch.sum(loss(torch.sign(true_vals)*preds, true_vals))/torch.sign(true_vals).sum())
                d.append(l.item())
            loss_["ts_"+str(epoch)]= np.mean(d)
            loss_["preds_"+str(epoch)] = preds.numpy()
            loss_["true_" + str(epoch)] = true_vals.numpy()
    return loss_

if __name__ == "__main__":

    batch_size = 256

    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(df, num_users, num_items)
    _, _, _, train_inter_mat = load_data_ml100k(train_data, num_users, num_items)
    _, _, _, test_inter_mat = load_data_ml100k(test_data, num_users, num_items)
    train_inter_mat = train_inter_mat.astype(np.float32)
    test_inter_mat = test_inter_mat.astype(np.float32)
    train_iter = torch.utils.data.DataLoader(train_inter_mat, shuffle=True, batch_size=batch_size)
    test_iter = torch.utils.data.DataLoader(test_inter_mat, shuffle=False, batch_size=batch_size)

    num_hiddens = 500
    lr = 0.002
    num_epochs = 100
    wd = 1e-5
    net = AutoRec(num_hiddens=num_hiddens, num_users=num_users)
    loss = torch.nn.MSELoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    res = train_autorec(net, train_iter, test_iter, loss, trainer, num_epochs, devices="cuda:0")
    plt.plot([res["tr_" + str(epoch)] for epoch in range(num_epochs)], label="training loss")
    plt.plot([res["ts_" + str(epoch)] for epoch in range(num_epochs)], label="eval loss")
    plt.grid()
    plt.legend()