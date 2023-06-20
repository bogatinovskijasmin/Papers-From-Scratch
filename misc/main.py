import numpy as np
import matplotlib.pyplot as plt
import torch

from data import Data
from models import LinearRegressionScratch
from trainer import Trainer


if __name__ == "__main__":
    batch_size = 512
    num_inputs = 5
    max_epochs = 20
    gradient_clip_val = 0
    lr = 0.1
    sigma = 0.01
    momentum = 0.8
    beta = 0.8
    beta2 = 0.99

    data = Data(batch_size=batch_size)
    model = LinearRegressionScratch(d_input=num_inputs, lr=lr, momentum=momentum, beta=beta, beta2=beta2)
    trainer = Trainer(max_epochs=max_epochs, gradient_clip_val=gradient_clip_val)
    trainer.fit(data=data, model=model)


    s = [trainer.training_loss[idx] for idx in range(trainer.max_epochs)]
    plt.plot(s, label="training loss")
    t = [trainer.validation_loss[idx] for idx in range(trainer.max_epochs)]
    plt.plot(t, label="validation loss")
    plt.legend()
    plt.show()
