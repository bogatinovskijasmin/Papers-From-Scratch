import torch
from main_classes import Encoder, Decoder, EncoderDecoder
from attention_functions import AddetiveAttention

class SeqEncoder(Encoder):
    def __init__(self, vocab_size, input_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, input_size)
        self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X.t())
        rnn_output, states = self.rnn(X)
        return rnn_output, states


class SeqDecoderAttention(Decoder):
    def __init__(self, vocab_size, input_size, num_hiddens, num_layers, dropout, use_bias):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.input_size = input_size
        self.num_layers = num_layers

        self.attention = AddetiveAttention(num_hiddens=num_hiddens, dropout=dropout, bias=use_bias)
        self.embedding = torch.nn.Embedding(vocab_size, input_size)
        self.rnn = torch.nn.GRU(input_size=input_size+num_hiddens, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout)
        self.output_layer = torch.nn.LazyLinear(vocab_size)

    def init_states(self, X, valid_seq_lens, *args):
        rnn_out, hiddens = X[0], X[1]
        return (rnn_out.permute(1, 0, 2), hiddens, valid_seq_lens)

    def forward(self, X, states, *args):
        # print("X.shape", X.shape)
        X_dec = self.embedding(X)
        enc_states, hiddens, valid_seq_lens = states
        output = []
        # print("X_dec.shape", X_dec.shape)
        X_dec = X_dec.permute(1, 0, 2)
        # print("X_dec.shape", X_dec.shape)

        for x in X_dec:
            query = hiddens[-1]
            # print("query.shape", query.shape)
            query = query.unsqueeze(1)
            # print("query.shape", query.shape)
            attention = self.attention(query, enc_states, enc_states, valid_seq_lens)
            x = x.unsqueeze(0)
            x = x.permute(1, 0, 2)

            # print(f"attention.shape {attention.shape}")
            # print(f"x.shape {x.shape}")

            context_and_input = torch.cat([attention, x], dim=-1)
            # print(f"1. context_and_input.shape {context_and_input.shape}")
            context_and_input = context_and_input.permute(1, 0, 2)
            # print(f"2. context_and_input.shape {context_and_input.shape}")
            dec_output, hiddens = self.rnn(context_and_input, hiddens)
            # print(f"dec_output.shape", dec_output.shape)
            output.append(dec_output)
            # print("========"*10)
        out = torch.cat(output, dim=0)
        # print(f"out.shape ", out.shape)
        out = self.output_layer(out).permute(1, 0, 2)
        # print(f"out.shape ", out.shape)
        return out, [enc_states, hiddens, valid_seq_lens]

class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, pad_token, lr=0.01):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)
        self.pad_token = pad_token
    def loss(self, y_pred, y, averaged=True):
        loss = super(Seq2Seq, self).loss(y_pred, y=y, averaged=False)
        mask = 1.*(y.reshape(-1)!=self.pad_token)
        loss = (loss*mask).sum()/mask.sum()
        return loss