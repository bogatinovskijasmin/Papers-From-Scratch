import torch
from utils import Adagrad_Scratch, SGD_Scratch

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def predict(self):
        raise NotImplementedError
    def loss(self):
        raise NotImplementedError
    def forward(self):
        raise NotImplementedError
    def configure_optimizer(self):
        raise NotImplementedError
    def train_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        acc = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l, acc
    def valid_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        acc = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l, acc
    def accuracy_score(self, y_pred, y, averaged=True):
        y_pred = torch.argmax(y_pred, axis=1).squeeze()
        y_pred = y_pred.type(y.dtype)
        compared = 1.*(y_pred==y)
        return compared.mean() if averaged else compared
    def checkpoint(self, checkpoint_name='default.pth'):
        torch.save(self.state_dict(), checkpoint_name)
    def load_check(self, checkpoint_name='default.pth'):
        self.load_state_dict(torch.load(checkpoint_name))

class SoftaxRegressionScratch(Classifier):
    def __init__(self, n_inputs, n_hidden, output, dropout_rate1, hyperparameters):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = output
        self.lr = hyperparameters['lr']
        self.eta = hyperparameters['eta']
        self.dropout_rate1 = dropout_rate1
        self.W1 = torch.nn.parameter.Parameter(torch.normal(0, 0.01, (n_inputs, n_hidden), requires_grad=True))
        torch.nn.init.xavier_uniform_(self.W1)
        self.b1 = torch.nn.parameter.Parameter(torch.zeros((n_hidden), requires_grad=True))
        self.W2 = torch.nn.parameter.Parameter(torch.normal(0, 0.01, (n_hidden, output), requires_grad=True))
        torch.nn.init.xavier_uniform_(self.W2)
        self.b2 = torch.nn.parameter.Parameter(torch.zeros((output), requires_grad=True))

    def ReLU(self, X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    def dropout_layer(self, X, dropout_rate):
        assert 0 <= dropout_rate <= 1
        if dropout_rate==1: return torch.zeros_like(X, device="cuda:0")
        dropout_rate = torch.tensor(dropout_rate, device="cuda:0")
        mask = torch.tensor(1., device="cuda:0")*(torch.rand(X.shape, device="cuda:0")>dropout_rate)
        return X*mask/(1-dropout_rate)
    def softmax(self, y_pred):
        exp_y = torch.exp(y_pred)
        denominator = exp_y.sum(axis=1, keepdims=True)
        return exp_y/denominator
    def forward(self, X):
        X = X.reshape((-1, self.n_inputs))
        output1 = self.ReLU(torch.matmul(X, self.W1) + self.b1)
        if self.training:
            output1 = self.dropout_layer(output1, dropout_rate=self.dropout_rate1)
        # return torch.nn.functional.softmax(torch.matmul(output1, self.W2) + self.b2, dim=1)
        return self.softmax(torch.matmul(output1, self.W2) + self.b2)
    def loss(self, softmax_logits, y):
        return -torch.log(softmax_logits[list(range(len(y))), y]).mean()
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
    def configure_optimizer(self):
        return Adagrad_Scratch(self.parameters(), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

class SoftmaxRegression(Classifier):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout_rate, hyperparameters):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate
        self.lr = hyperparameters['lr']
        self.eta = hyperparameters['eta']
        self.net = torch.nn.Sequential(torch.nn.Flatten(),
                                       torch.nn.Linear(n_inputs, n_hiddens),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(n_hiddens, n_outputs)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.net:
            if type(module) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    def forward(self, X):
        return self.net(X)
    def loss(self, y_pred, y):
        return torch.nn.functional.cross_entropy(y_pred, y)
    def configure_optimizer(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)
        # return Adagrad_Scratch(self.parameters(), lr=self.lr)