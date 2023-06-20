import torch
from collections import defaultdict

class Trainer(torch.nn.Module):
    def __init__(self, max_epochs, lr_scheduler=None):
        super().__init__()
        self.max_epochs = max_epochs
        self.num_gpus = [f"cuda:{x}" for x in range(torch.cuda.device_count())]
        self.scheduler = lr_scheduler

        self.training_loss_res = defaultdict(list)
        self.training_acc_res = defaultdict(list)
        self.validation_loss_res = defaultdict(list)
        self.validation_acc_res = defaultdict(list)

    def prepare_batch(self, batch):
        if self.num_gpus:
            return [a.to(self.num_gpus[0]) for a in batch]
        return batch
    def prepare_data(self, data):
        self.training_dataloader = data.train_dataloader()
        self.validation_dataloader = data.val_dataloader()
        self.training_batches_count = len(self.training_dataloader)
        self.validation_batches_count = len(self.validation_dataloader)
    def prepare_model(self, model):
        self.model = model
        if self.num_gpus:
            self.model.to(self.num_gpus[0])
        self.trainer = self
    def fit(self, data, model):
        self.prepare_data(data=data)
        self.prepare_model(model=model)
        self.optim = self.model.configure_optimizer()
        self.training_batch_id = 0
        self.validation_batch_id = 0
        self.epoch = 0
        for self.epoch in range(self.max_epochs):
            print(f"Current epoch run {self.epoch + 1}")
            self.run_epoch()
    def run_epoch(self):
        self.model.train()
        for batch in self.training_dataloader:
            l, acc = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                l.backward()
                self.optim.step()
            self.training_batch_id+=1
            self.training_loss_res[self.epoch].append(torch.exp(l))
            self.training_acc_res[self.epoch].append(acc)
        self.model.eval()
        for batch in self.validation_dataloader:
            l, acc = self.model.training_step(self.prepare_batch(batch))
            self.validation_loss_res[self.epoch].append(torch.exp(l))
            self.validation_acc_res[self.epoch].append(acc)
            self.validation_batch_id += 1
