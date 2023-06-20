import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim.lr_scheduler

from utils import FashionMNIST, Timer, ConstScheduler
from classifiers import SoftaxRegressionScratch, SoftmaxRegression
from trainer import Trainer


batch_size = 256
max_epochs = 10
num_inputs = 784
num_hiddens = 256
num_outputs = 10
dropout_rate = 0.01
# hyperparameters = {"lr":0.01}
hyperparameters = {"lr":1}
scheduler = ConstScheduler(init_lr=hyperparameters["lr"])
schedulerPyTorch = torch.optim.lr_scheduler.MultiStepLR
timer = Timer()

if __name__ == "__main__":

    # data = FashionMNIST(batch_size=batch_size)
    #
    # trainer = Trainer(max_epochs=max_epochs, scheduler=scheduler)
    # model = SoftaxRegressionScratch(n_inputs=num_inputs,
    #                                 output=num_outputs,
    #                                 n_hidden=num_hiddens,
    #                                 hyperparameters=hyperparameters,
    #                                 dropout_rate1=dropout_rate)
    # timer.start()
    # trainer.fit(data=data, model=model)
    # timer.stop()
    #
    #
    # plt.subplot(211)
    # plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_loss).mean(axis=0), label="training loss", linestyle="--", color="blue")
    # plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_loss).mean(axis=0), label="validation loss",
    #          linestyle="--", color="red")
    #
    # plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.training_acc).mean(axis=0), label="training loss",
    #          linestyle="-", color="blue")
    # plt.plot(np.arange(max_epochs), pd.DataFrame(trainer.val_acc).mean(axis=0), label="validation acc",
    #          linestyle="-", color="red")
    # plt.grid()
    # plt.legend()


    data = FashionMNIST(batch_size=batch_size)
    trainer = Trainer(max_epochs=max_epochs, scheduler_fcn=schedulerPyTorch)
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