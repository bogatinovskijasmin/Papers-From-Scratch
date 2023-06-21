import torch

from attention_networks import AddNorm, PositionalEncoding, FeedForward
from attention_functions import MultiHeadAttention

class AttentionEncoderBlock(torch.nn.Module):
    def __init__(self, num_hiddens, dropout, num_heads, bias, ffn_input):
        super().__init__()
        self.num_heads =num_heads
        self.bias = bias
        self.add_norm1 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.add_norm2 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.attention = MultiHeadAttention(dropout=dropout, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(ffn_input=ffn_input, ffn_output=num_hiddens)
    def forward(self, X, valid_seq_len, *args):
        # X is of shape (batch_size, # queries, #query-size)
        attention = self.attention(X, X, X, valid_seq_len)
        X = self.add_norm1(X, attention)
        return  self.add_norm2(X, self.ffn(X))

class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers,  num_hiddens, dropout, num_heads, bias, ffn_input, vocab_size):
        super().__init__()
        self.num_layers = num_layers
        self.blks = torch.nn.Sequential()
        self.embedding = torch.nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens=num_hiddens, dropout=dropout, max_len=1000)
        for i in range(self.num_layers):
            self.blks.add_module(f"block:{i}", AttentionEncoderBlock(num_hiddens=num_hiddens, dropout=dropout, num_heads=num_heads, bias=bias, ffn_input=ffn_input))
    def forward(self, X, valid_seq_len, *args):
        X = self.pos_encoding(X)
        for i, blk in enumerate(self.blks):
            X = blk(X=X, valid_seq_len=valid_seq_len, *args)
        return X



