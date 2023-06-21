import torch
class AddNorm(torch.nn.Module):
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.batch_norm = torch.nn.LayerNorm(num_hiddens)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, X, Y):
        return self.batch_norm(X + self.dropout(Y))

    def __call__(self, X, Y):
        return self.forward(X, Y)
class FeedForward(torch.nn.Module):
    def __init__(self, ffn_input, ffn_output):
        super().__init__()
        self.dense1 = torch.nn.LazyLinear(ffn_input)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.LazyLinear(ffn_output)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

    def __call__(self, X):
        return self.forward(X)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, num_hiddens, dropout):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len).reshape(-1, 1)/torch.pow(10000, torch.arange(0, num_hiddens, 2)/num_hiddens)
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)
        self.dropout = torch.nn.Dropout(dropout)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :].to(X.device))