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



class GRU(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.b_fg = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W_fg = torch.nn.Parameter(torch.normal(0, self.sigma, (num_hiddens, num_hiddens)))
        self.W_fx = torch.nn.Parameter(torch.normal(0, self.sigma, (num_inputs, num_hiddens)))

        self.b_ig = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W_ig = torch.nn.Parameter(torch.normal(0, self.sigma, (num_hiddens, num_hiddens)))
        self.W_ix = torch.nn.Parameter(torch.normal(0, self.sigma, (num_inputs, num_hiddens)))

        self.b_ug = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W_ug = torch.nn.Parameter(torch.normal(0, self.sigma, (num_hiddens, num_hiddens)))
        self.W_ux = torch.nn.Parameter(torch.normal(0, self.sigma, (num_inputs, num_hiddens)))

        self.b_og = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W_og = torch.nn.Parameter(torch.normal(0, self.sigma, (num_hiddens, num_hiddens)))
        self.W_ox = torch.nn.Parameter(torch.normal(0, self.sigma, (num_inputs, num_hiddens)))


    def forward(self, X, state=None):
        # X (num_steps, batch_size, num_inputs)
        rnn_outputs = []
        if state is None:
            hidden_state = torch.zeros((X.shape[1], self.num_hiddens)).to(X.device)
        else:
            hidden_state  = state

        for i in range(X.shape[0]):
            reset_rule = torch.sigmoid(torch.matmul(X[i], self.W_fx) + torch.matmul(hidden_state, self.W_fg + self.b_fg))
            update_rule = torch.sigmoid(torch.matmul(X[i], self.W_ix) + torch.matmul(hidden_state, self.W_ig + self.b_ig))
            internal_state_rule = torch.tanh(torch.matmul(X[i], self.W_ux) + torch.matmul(reset_rule*hidden_state, self.W_ug + self.b_ug))
            hidden_state = update_rule*hidden_state + (1-update_rule)*internal_state_rule
            rnn_outputs.append(hidden_state)

        rnn_outputs = torch.cat(rnn_outputs, dim=0).reshape(X.shape[0], X.shape[1], -1)
        # print(f"rnn.output {rnn_outputs.shape}")
        return rnn_outputs, hidden_state

class StackedRNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma, num_blocks):
        super().__init__()
        self.rnn = torch.nn.Sequential()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.rnn.add_module(f"block:{i}", GRU(num_inputs=num_inputs, num_hiddens=num_hiddens, sigma=sigma) if i==0 else GRU(num_inputs=num_hiddens, num_hiddens=num_hiddens, sigma=sigma))

    def forward(self, X, state=None):
        if state is None: state = [None]*self.num_blocks
        for i in range(self.num_blocks):
            X, state[i] = self.rnn[i].forward(X, state[i])
        return X, state

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

