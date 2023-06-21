import torch

class Classifier(torch.nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    def training_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        # with torch.no_grad():
        #     a = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l
    def validation_step(self, batch):
        l = self.loss(self.forward(*batch[:-1]), batch[-1])
        # with torch.no_grad():
        #     a = self.accuracy_score(self.forward(*batch[:-1]), batch[-1])
        return l
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

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def init_states(self, X, *args):
        raise NotImplementedError
    def forward(self, X, *args):
        raise NotImplementedError

class EncoderDecoder(Classifier):
    def __init__(self, encoder, decoder, lr):
        super().__init__(lr=lr)
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
    def forward(self, X_enc, X_dec, *args):
        encoder_out = self.encoder(X_enc, *args)
        states = self.decoder.init_states(encoder_out, *args)
        output = self.decoder(X_dec, states, *args)
        return output