import torch
from d2l import torch as d2l
from collections import defaultdict

class Trainer(d2l.Trainer):
    def __init__(self, max_epochs):
        super().__init__(max_epochs=max_epochs)
        self.max_epochs = max_epochs
        self.acc_val = defaultdict(list)
        self.loss_val = defaultdict(list)
        self.loss_train = defaultdict(list)
        self.gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    def prepare_data(self, data):
        self.train_dataloader = data.get_traindataloader()
        self.val_dataloader = data.get_valdataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(self.val_dataloader)
    def prepare_model(self, model):
        model.trainer = self
        self.model = model

        if self.gpus:
            self.model.to(self.gpus[0])

    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizer()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            print("Running Epoch {}".format(self.epoch))
            self.fit_epoch()
    def prepare_batch(self, batch):
        if self.gpus:
            return [a.to(self.gpus[0]) for a in batch]
        return batch
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                # for param in self.model.parameters():
                #     print(param.grad)
                self.optim.step()
            self.loss_train[self.epoch].append(loss.item())

        self.model.eval()
        for batch in self.val_dataloader:
            l_val, acc_val = self.model.validation_step(self.prepare_batch(batch))
            self.acc_val[self.epoch].append(acc_val)
            self.loss_val[self.epoch].append(l_val.item())
