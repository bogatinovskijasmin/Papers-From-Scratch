import torch
class Classifier(torch.nn.Module):
    def __init__(self, lr=0.1, lambd=0.01):
        super().__init__()
        self.lr = lr
        self.lambd = lambd
    def forward(self):
        raise NotImplementedError
    def training_step(self, batch):
        loss = self.loss(self.forward(*batch[:-1]), batch[-1])
        return loss
    def validation_step(self, batch):
        y_hat = self.forward(*batch[:-1])
        loss = self.loss(y_hat, batch[-1])
        acc = self.accuracy(y_hat, batch[-1])
        return loss, acc
    def accuracy(self, y_hat, y, averaged=True):
        y_hat = y_hat.argmax(axis=1).type(y.dtype)
        compared = 1.*(y_hat.squeeze()==y.squeeze())
        return compared.mean() if averaged else compared
    def configure_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def loss(self, y_pred, y, averaged=True):
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[-1]))
        y = torch.reshape(y, (-1, ))
        return torch.nn.functional.cross_entropy(y_pred, y, reduction="mean" if averaged else "none")