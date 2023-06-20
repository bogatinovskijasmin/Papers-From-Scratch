import torch
import matplotlib.pyplot as plt
import numpy as np

from models import RNN, LanguageModel
from utils import TimeMachine
from trainer import Trainer


if __name__=="__main__":
    batch_size = 128
    hiddens = 32
    num_steps = 9
    max_epochs = 5
    embed_size = 32
    sigma = 0.01

    lr = 0.05

    data = TimeMachine(batch_size=batch_size, num_steps=num_steps)
    vocab_size = int(len(data.vocab.idx_to_token))
    rnn_model = RNN(num_inputs=embed_size, num_hiddens=hiddens, sigma=sigma)

    rnn_model = torch.nn.RNN(embed_size, hiddens)
    # rnn_model = RNN(num_inputs=embed_size, num_hiddens=hiddens, sigma=sigma)
    

    model = LanguageModel(rnn=rnn_model, num_hiddens=hiddens, sigma=sigma, vocab_size=vocab_size, lr=lr)
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(data=data, model=model)

    z = [np.mean(trainer.training_loss_res[idx][idj].cpu().item()) for idx in range(5) for idj in range(len(trainer.training_loss_res[idx]))]
    plt.plot(z, label="training loss")
    z = [np.mean(trainer.validation_loss_res[idx][idj].cpu().item()) for idx in range(5) for idj in range(len(trainer.validation_loss_res[idx]))]
    plt.plot(z, label="validation loss")
    plt.legend()


