import torch

from main_classes import EncoderDecoder
from attention_networks import AddNorm, PositionalEncoding, FeedForward
from attention_functions import MultiHeadAttention

class AttentionEncoderBlock(torch.nn.Module):
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
        print("X.encoder.shape ", X.shape)
        attention = self.attention(X, X, X, valid_seq_len)
        print("attention.encoder.shape ", attention.shape)
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
        X = self.embedding(X)
        X = self.pos_encoding(X)
        for i, blk in enumerate(self.blks):
            X = blk(X=X, valid_seq_len=valid_seq_len, *args)
        return X
class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, num_hiddens, dropout, num_heads, bias, ffn_input, i):
        super().__init__()
        self.i = i
        self.add_norm1 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.add_norm2 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.add_norm3 = AddNorm(num_hiddens=num_hiddens, dropout=dropout)
        self.ffn1 = FeedForward(ffn_input=ffn_input, ffn_output=num_hiddens)
        self.attention1 = MultiHeadAttention(dropout=dropout, num_hiddens=num_hiddens, num_heads=num_heads, bias=bias)
        self.attention2 = MultiHeadAttention(dropout=dropout, num_hiddens=num_hiddens, num_heads=num_heads, bias=bias)
    def forward(self, X, states):
        enc_out, valid_enc_len = states[0], states[1]

        if states[2][self.i] is None:
            decoder_query = X
        else:
            decoder_query = torch.cat([states[2][self.i], X], dim=1)

        states[2][self.i] = decoder_query

        if self.training:
            batch_size, num_steps, _ = X.shape
            valid_dec_len = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            valid_dec_len = None

        Y = self.attention1(X, decoder_query, decoder_query, valid_dec_len)
        X = self.add_norm1(X, Y)
        Y = self.attention2(X, enc_out, enc_out, valid_enc_len)
        X = self.add_norm2(X, Y)
        Y = self.add_norm3(X, self.ffn1(X))
        return Y, states

class TransformerDecoder(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, num_hiddens, dropout, num_heads, bias, ffn_input):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding = PositionalEncoding(num_hiddens=num_hiddens, max_len=1000, dropout=dropout)
        self.decoders = torch.nn.Sequential()
        for i in range(self.num_layers):
            self.decoders.add_module(f"block{i}", TransformerDecoderBlock(num_hiddens=num_hiddens, dropout=dropout, num_heads=num_heads, bias=bias, ffn_input=ffn_input, i=i))
        self.output_layer = torch.nn.LazyLinear(vocab_size)
    def init_states(self, X, valid_seq_len, *args):
        return [X, valid_seq_len, [None]*self.num_layers]
    def forward(self, X, states, *args):
        X = self.embedding(X)
        X = self.pos_embedding(X)
        for i, dec in enumerate(self.decoders):
            X, states = dec(X, states)
        out = self.output_layer(X)
        return out, states


class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, pad_token, lr=0.01):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)
        self.pad_token = pad_token
    def loss(self, y_pred, y, averaged=True):
        loss = super(Seq2Seq, self).loss(y_pred, y=y, averaged=False)
        mask = 1.*(y.reshape(-1)!=self.pad_token)
        loss = (loss*mask).sum()/mask.sum()
        return loss