import torch
from main_classes import Encoder, Decoder, EncoderDecoder



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


class SeqDecoder(Decoder):
    def __init__(self, vocab_size, input_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.input_size = input_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, input_size)
        self.rnn = torch.nn.GRU(input_size=input_size+num_hiddens, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout)
        self.output_layer = torch.nn.LazyLinear(vocab_size)

    def init_states(self, X, *args):
        rnn_out, hiddens = X[0], X[1]
        return (rnn_out, hiddens)
    def forward(self, X, states, *args):
        X_dec = self.embedding(X.t())
        enc_states, hiddens = states
        output = []
        context = hiddens[-1]
        context = context.unsqueeze(0)
        for x in X_dec:
            x = x.unsqueeze(0)
            context_and_input = torch.cat([context, x], dim=-1)
            dec_output, hiddens = self.rnn(context_and_input, hiddens)
            output.append(dec_output)
        dec_output = torch.cat(output, dim=0)
        out = self.output_layer(dec_output).permute(1, 0, 2)
        return out, [dec_output, hiddens]

class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, pad_token, lr=0.01):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)
        self.pad_token = pad_token
    def loss(self, y_pred, y, averaged=True):
        loss = super(Seq2Seq, self).loss(y_pred, y=y, averaged=False)
        mask = 1.*(y.reshape(-1)!=self.pad_token)
        loss = (loss*mask).sum()/mask.sum()
        return loss