import torch

from main_classes import EncoderDecoder
from attention_networks import AddNorm, PositionalEncoding, FeedForward
from attention_functions import MultiHeadAttention

class BERTEncoderBlock(torch.nn.Module):
    def __init__(self, num_hiddens, dropout, num_heads, bias, ffn_input):
        super().__init__()
        self.num_heads =num_heads
        self.bias = bias
        self.add_norm1 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.add_norm2 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.attention = MultiHeadAttention(dropout=dropout, num_hiddens=num_hiddens, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(ffn_input=ffn_input, ffn_output=num_hiddens)
    def forward(self, X, valid_seq_len, *args):
        # X is of shape (batch_size, # queries, #query-size)
        attention = self.attention(X, X, X, valid_seq_len)
        X = self.add_norm1(X, attention)
        return  self.add_norm2(X, self.ffn(X))

class BERTEncoder(torch.nn.Module):
    def __init__(self, num_layers,  num_hiddens, dropout, num_heads, bias, ffn_input, vocab_size):
        super().__init__()
        self.num_layers = num_layers
        max_len = 1000

        self.blks = torch.nn.Sequential()
        self.token_embedding = torch.nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = torch.nn.Embedding(2, num_hiddens)
        self.pos_encoding = torch.nn.Parameter(torch.randn((1, max_len, num_hiddens)))

        for i in range(self.num_layers):
            self.blks.add_module(f"block:{i}", BERTEncoderBlock(num_hiddens=num_hiddens, dropout=dropout, num_heads=num_heads, bias=bias, ffn_input=ffn_input))
    def forward(self, X, segment, valid_seq_len, *args):
        X = self.token_embedding(X)
        segment = self.segment_embedding(segment)
        X = X + segment + self.pos_encoding[:, :X.shape[1], :]

        for i, blk in enumerate(self.blks):
            X = blk(X=X, valid_seq_len=valid_seq_len, *args)
        return X

    def __call__(self, X, segment, valid_seq_len, *args):
        return self.forward(X, segment, valid_seq_len, *args)

class BERT(torch.nn.Module):
    def __init__(self, num_layers,  num_hiddens, dropout, num_heads, bias, ffn_input, vocab_size):
        super().__init__()
        self.mlm = MaskLanguageModeling(num_hiddens=num_hiddens, vocab_size=vocab_size)
        self.nsp = NextSentancePrediction(num_hiddens)
        self.encoder = BERTEncoder(num_layers=num_layers,  num_hiddens=num_hiddens, dropout=dropout, num_heads=num_heads, bias=bias, ffn_input=ffn_input, vocab_size=vocab_size)

    def forward(self, X, segment, valdid_seq_len=None, pred_pos=None, *args):
        X_encoded = self.encoder(X, segment, valdid_seq_len)
        nsp = self.nsp.forward(X_encoded[:, 0, :])
        if pred_pos is not None:
            mlm = self.mlm.forward(X_encoded, pred_pos)
        else:
            mlm = None
        return X_encoded, mlm, nsp


class MaskLanguageModeling(torch.nn.Module):
    def __init__(self, num_hiddens, vocab_size):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LazyLinear(num_hiddens),
                                 torch.nn.ReLU(),
                                 torch.nn.LayerNorm(num_hiddens),
                                 torch.nn.LazyLinear(vocab_size))

    def forward(self, X, masked_positions, *args):

        num_masked_positions_to_predict = masked_positions.shape[1]
        pred_positions = masked_positions.reshape(-1)

        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_masked_positions_to_predict)

        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_masked_positions_to_predict, -1))

        mlm_y_hat = self.net(masked_X)
        return mlm_y_hat



class NextSentancePrediction(torch.nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.dense_layer1 = torch.nn.LazyLinear(num_hiddens)
        self.tanh = torch.nn.Tanh()
        self.dense_layer2 = torch.nn.LazyLinear(2)
    def forward(self, X):
        return self.dense_layer2(self.tanh(self.dense_layer1(X)))