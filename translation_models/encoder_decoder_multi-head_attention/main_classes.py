import collections

import torch
import math
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
        return output[0]

    def predict_step(self, batch, device, num_steps,
                     save_attention_weights=False):
        batch = [a.to(device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_states(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [tgt[:, (0)].unsqueeze(1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(2))
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.cat(outputs[1:], 1), attention_weights


    def bleu(self, prediction_seq, true_pred, k=4):
        pred_seq, true_seq = prediction_seq.split(" "), true_pred.split(" ")
        len_pred_seq, len_true_seq = len(pred_seq), len(true_seq)
        score = math.exp(min(0, 1-len_true_seq/len_pred_seq))

        for n in range(1, min(k, len_pred_seq)+1):

            existing_n_grams = collections.defaultdict(int)
            num_matches = 0
            for i in range(len_true_seq-n+1):
                existing_n_grams[" ".join(true_seq[i:i+n])] +=1

            for i in range(len_pred_seq-n+1):
                if existing_n_grams[" ".join(true_seq[i:i+n])]>0:
                    existing_n_grams[" ".join(true_seq[i:i+n])] -= 1
                    num_matches +=1
            score *= math.pow(num_matches/(len_pred_seq-n+1), math.pow(0.5, n))

        return score