import torch
from collections import defaultdict

class Trainer:
    def __init__(self, max_epochs):
        # super().__init__(max_epochs=max_epochs)
        self.max_epochs = max_epochs
        self.gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.training_loss = defaultdict(list)
        self.training_acc = defaultdict(list)
        self.val_loss = defaultdict(list)
        self.val_acc = defaultdict(list)
    def prepare_data(self, data):
        self.training_dataloader = data.get_traindataloader()
        self.val_dataloader = data.get_valdataloader()
        self.num_training_batches = len(self.training_dataloader)
        self.num_validation_batches = len(self.val_dataloader)
    def prepare_model(self, model):
        model.trainer = self
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model
    def prepare_batch(self, batch):
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]
        return batch
    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizer()
        self.epoch = 0
        self.training_batch_idx = 0
        self.validation_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            print(f"executing epoch {self.epoch+1}")
            self.fit_epoch()

    def cuda2cpu(self, x):
        return x.item()
    def fit_epoch(self):
        self.model.train()
        for batch in self.training_dataloader:
            l, acc = self.model.train_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                l.backward()
                self.optim.step()
            self.training_batch_idx+=1
            self.training_loss[self.epoch].append(self.cuda2cpu(l))
            self.training_acc[self.epoch].append(self.cuda2cpu(acc))
        self.model.eval()
        for batch in self.val_dataloader:
            l, acc = self.model.valid_step(self.prepare_batch(batch))
            self.val_loss[self.epoch].append(self.cuda2cpu(l))
            self.val_acc[self.epoch].append(self.cuda2cpu(acc))