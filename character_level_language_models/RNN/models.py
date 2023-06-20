import torch

class Classifier(torch.nn.Module):

    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        with torch.no_grad():
            a = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l, a

    def validation_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        with torch.no_grad():
            a = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l, a

    def accuracy_score(self, y_pred, y, averaged=True):
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[-1])).argmax(dim=-1)
        y = torch.reshape(y, (-1,))
        compare = 1.*(y_pred==y)
        return torch.mean(compare) if averaged else compare

    def loss(self, y_pred, y, averaged=True):
        # print(f"y shape {y.shape}, y_pred shape {y_pred.shape}")
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[-1]))
        y = torch.reshape(y, (-1, ))
        # print(f"y shape {y.shape}, y_pred shape {y_pred.shape}")
        return torch.nn.functional.cross_entropy(y_pred, y, reduction="mean" if averaged else "none")


class RNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.W_h = torch.nn.Parameter(torch.normal(0, self.sigma, (num_hiddens, num_hiddens)))
        self.b_h = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W_x = torch.nn.Parameter(torch.normal(0, self.sigma, (num_inputs, num_hiddens)))

    def forward(self, X, state=None):
        # X (num_steps, batch_size, num_inputs)
        rnn_outputs = []
        if state is None:
            state = torch.zeros((X.shape[1], self.num_hiddens)).to(X.device)
        else:
            state = state

        for i in range(X.shape[0]):
            rnn_outputs.append(torch.tanh(torch.matmul(X[i], self.W_x) + torch.matmul(state, self.W_h) + self.b_h))

        rnn_outputs = torch.cat(rnn_outputs, dim=0).reshape(X.shape[0], X.shape[1], -1)
        # print(f"rnn_outputs {rnn_outputs.shape}")
        return rnn_outputs, state

class LanguageModel(Classifier):
    def __init__(self, rnn, num_hiddens, vocab_size, sigma, lr=0.1):
        super().__init__(lr=lr)
        self.embedding = torch.nn.Embedding(vocab_size, num_hiddens)
        self.rnn = rnn
        self.num_outputs = self.vocab_size = vocab_size
        self.sigma = sigma
        self.num_hiddens = num_hiddens
        self.output_layer()


    def output_layer(self):
        self.W_o = torch.nn.Parameter(torch.normal(0, self.sigma, (self.num_hiddens, self.num_outputs)))
        self.b_o = torch.nn.Parameter(torch.zeros(self.num_outputs))
        # self.linear = torch.nn.LazyLinear(self.num_outputs)

    def output_layer_pred(self, X):
        return torch.matmul(X, self.W_o) + self.b_o

    def forward(self, X, states=None):
        # X (batch_size, num_steps)
        # X.t() (num_steps, batch_size)
        X = self.embedding(X.t())
        # X (num_steps, batch_size, embed_size)
        rnn_outputs, state = self.rnn(X)
        out = self.output_layer_pred(rnn_outputs)
        # print(f"the output is {out.shape}")
        return out

