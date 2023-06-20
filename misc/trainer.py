import torch
import collections

class Trainer:
    def __init__(self, max_epochs, gradient_clip_val, learning_scheduler=None, scheduler_params=None):
        self.max_epochs = max_epochs
        self.num_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.scheduler = learning_scheduler
        self.scheduler_params = scheduler_params
        self.training_loss = collections.defaultdict(list)
        self.validation_loss = collections.defaultdict(list)
        self.gradient_clip_val = gradient_clip_val
    def prepare_data(self, data):
        self.training_dataloader = data.get_traindataloader()
        self.validation_dataloader = data.get_valdataloader()
        self.num_training_batches = len(self.training_dataloader)
        self.num_val_batches = len(self.validation_dataloader)
    def prepare_model(self, model):
        self.model = model
        if len(self.num_gpus)>0:
            model.to(self.num_gpus[0])
        model.trainer = self
    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizer()
        self.training_idx = 0
        self.validation_idx = 0
        for self.epoch in range(self.max_epochs):
            self.run_epoch()
    def prepare_batch(self, batch):
        if len(self.num_gpus):
            batch = [a.to(self.num_gpus[0]) for a in batch]
        return batch
    def run_epoch(self):
        self.model.train()
        for batch in self.training_dataloader:

            l = self.model.training_step(self.prepare_batch(batch))

            with torch.no_grad():
                l.backward()
                if self.gradient_clip_val>0:
                    self.clip_gradients(self.model, self.gradient_clip_val)

                # self.optim.step(t=self.training_idx+1)
                self.optim.step()
                self.training_idx+=1

            self.optim.zero_grad()
            self.training_loss[self.epoch] = l.item()

        self.model.eval()
        for batch in self.validation_dataloader:
            with torch.no_grad():
                l = self.model.training_step(self.prepare_batch(batch))
                self.validation_loss[self.epoch] = l.item()

    def clip_gradients(self, model, gradient_clip_val):
        parameters = [param for param in model.parameters() if param.requires_grad==True]
        gradient_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in parameters))
        if gradient_norm > gradient_clip_val:
            for p in parameters:
                p.grad[:] *= gradient_clip_val/gradient_norm