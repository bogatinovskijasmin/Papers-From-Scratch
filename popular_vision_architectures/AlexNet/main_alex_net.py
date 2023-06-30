import torch
import numpy as np
import matplotlib.pyplot as plt


from trainer import Trainer
from utils import FashionMNIST
from alex_net import AlexNet, init_cnn

if __name__ == "__main__":
    batch_size = 256
    max_epochs = 20
    lr = 0.01

    data = FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer = Trainer(max_epochs=max_epochs)
    model = AlexNet(num_classes=10, lr=lr)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
    model.layer_summary([1, 1, 224, 224])
    trainer.fit(data=data, model=model)
    s = [np.mean(trainer.loss_train[idx]) for idx in range(max_epochs)]
    plt.plot(s, label="training loss")
    s = [np.mean(trainer.loss_val[idx]) for idx in range(max_epochs)]
    plt.plot(s, label="validation loss")
    s = [np.mean([trainer.acc_val[idx][idy].item() for idy in range(len(trainer.acc_val[idx]))]) for idx in range(max_epochs)]
    plt.plot(s, label="validation acc")
