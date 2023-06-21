import torch
from collections import defaultdict

class Trainer:
    def __init__(self, max_epochs, gradient_clip_val, scheduler=None, scheduler_parameters=None):
        self.max_epcohs = max_epochs
        self.num_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.training_loss = defaultdict(list)
        self.validation_loss = defaultdict(list)
        self.validation_acc = defaultdict(list)
        self.gradient_clip_val = gradient_clip_val
        self.scheduler = scheduler
        self.scheduler_parameters = scheduler_parameters
    def prepare_data(self, data):
        self.training_dataloader = data.get_traindataloader()
        self.validation_dataloader = data.get_validation_dataloader()
        self.num_training_batches = len(self.training_dataloader)
        self.num_validation_batches = len(self.validation_dataloader)
    def prepare_model(self, model):
        self.model = model
        if len(self.num_gpus)>0:
            self.model.to(self.num_gpus[0])
        model.trainer = self
    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizer()
        # self.scheduler_parameters["optimizer"] = self.optim
        # self.scheduler_ = self.scheduler(**self.scheduler_parameters)
        self.training_batch_idx = 0
        self.validation_batch_idx = 0
        for self.epoch in range(self.max_epcohs):
            print(f"Currently executing epoch {self.epoch+1}")
            self.run_epoch()
    def prepare_batch(self, batch):
        if self.num_gpus:
            return [a.to(self.num_gpus[0]) for a in batch]
        return batch

    def run_epoch(self):
        self.model.train()
        for batch in self.training_dataloader:
            l = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                l.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradient(self.gradient_clip_val, self.model)
                self.optim.step()
                self.training_loss[self.epoch].append(l.item())

        # self.scheduler_.step()
        # print(f">>> Epcoh {self.epoch+1} learning rate {self.optim.param_groups[0]['lr']}")

        self.model.eval()
        for batch in self.validation_dataloader:
            l = self.model.validation_step(self.prepare_batch(batch))
            self.validation_loss[self.epoch].append(l.item())

    def clip_gradient(self, gradeint_clip_val, model):
        param = [param for param in model.parameters() if param.requires_grad==True]
        gradient_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in param))
        if gradient_norm > gradeint_clip_val:
            for p in param:
                p.grad[:] *= self.gradient_clip_val/gradient_norm
