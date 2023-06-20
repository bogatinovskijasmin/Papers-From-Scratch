import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import FashionMNIST, Timer
from classifiers import SoftaxRegressionScratch, SoftmaxRegression
from trainer import Trainer

batch_size = 256
max_epochs = 10
num_inputs = 784
num_hiddens = 16
num_outputs = 10
dropout_rate = 0.01
# hyperparameters = {"lr":0.01}
hyperparameters = {"lr":0.1, "eta":10**(-10)}
timer = Timer()

if __name__ == "__main__":

    data = FashionMNIST(batch_size=batch_size)
    trainer = Trainer(max_epochs=max_epochs)
    model = SoftaxRegressionScratch(n_inputs=num_inputs,
                                    output=num_outputs,
                                    n_hidden=num_hiddens,
                                    hyperparameters=hyperparameters,
                                    dropout_rate1=dropout_rate)
    timer.start()
    trainer.fit(data=data, model=model)
    timer.stop()


    plt.subplot(211)
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_loss).mean(axis=0), label="training loss", linestyle="--", color="blue")
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_loss).mean(axis=0), label="validation loss",
             linestyle="--", color="red")

    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_acc).mean(axis=0), label="training acc",
             linestyle="-", color="blue")
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_acc).mean(axis=0), label="validation acc",
             linestyle="-", color="red")
    plt.grid()
    plt.legend()


    data = FashionMNIST(batch_size=batch_size)
    trainer = Trainer(max_epochs=max_epochs)
    model = SoftmaxRegression(n_inputs=num_inputs,
                            n_outputs=num_outputs,
                            n_hiddens=num_hiddens,
                            hyperparameters=hyperparameters,
                            dropout_rate=dropout_rate)
    timer.start()
    trainer.fit(data=data, model=model)
    timer.stop()

    plt.subplot(212)
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_loss).mean(axis=0), label="training loss",
             linestyle="--", color="blue")
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_loss).mean(axis=0), label="validation loss",
             linestyle="--", color="red")

    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_acc).mean(axis=0), label="training loss",
             linestyle="-", color="blue")
    plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_acc).mean(axis=0), label="validation acc",
             linestyle="-", color="red")
    plt.grid()
    plt.legend()