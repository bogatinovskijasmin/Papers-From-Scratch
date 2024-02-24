from imports import *
import matplotlib.pyplot as plt
from PIL import Image
import os


class Trainer:
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.gpus = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
        self.test_dataloader = data.get_test_dataloader()

    def prepare_model(self, model):
        self.model = model
        if len(self.gpus) > 0:
            self.model.to(self.gpus[0])
        model.trainer = self

    def prepare_batch(self, batch):
        X, y = batch[0], batch[1]
        X = X.type(torch.float32).unsqueeze(1)
        X = X.to(self.gpus[0])
        y = y.to(self.gpus[0])
        return [X, y]

    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model=model)

        init_batch = next(iter(self.train_dataloader))
        
        init_batch = next(iter(data.get_test_dataloader()))
        init_batch = self.prepare_batch(init_batch)

        self.model.init_network(init_batch[0])
        self.optim = self.model.configure_optimizer()


        self.training_loss = {}
        self.valid_loss = {}
        self.valid_acc = {}

        for self.epoch in range(self.max_epochs):
            print(f"currently running {self.epoch+1}")
            self.fit_epoch()
        
    def fit_epoch(self):
        self.model.train()
        train_loss = []
        for step, batch in enumerate(self.train_dataloader):
            batch = self.prepare_batch(batch=batch)
            self.optim.zero_grad()
            l, acc = self.model.train_step(batch)
            l.backward()
            self.optim.step()
            to_save = l.detach().cpu().item()
            train_loss.append(to_save)
        self.training_loss[self.epoch] = torch.mean(torch.tensor(train_loss))

        self.model.eval()
        with torch.no_grad():
            eval_loss = []
            eval_acc = []
            for step, batch in enumerate(self.train_dataloader):
                batch = self.prepare_batch(batch=batch)
                l, acc = self.model.train_step(batch)
                to_save = l.detach().cpu().item()
                acc = acc.detach().cpu().item()
                eval_loss.append(to_save)
                eval_acc.append(acc)
        self.valid_loss[self.epoch] =  torch.mean(torch.tensor(eval_loss))
        self.valid_acc[self.epoch] = torch.mean(torch.tensor(eval_acc))


    def plot_curves(self, ):
        train_epochs, train_values = list(self.training_loss.keys()), list(self.training_loss.values())
        valid_epochs, valid_loss = list(self.valid_loss.keys()), list(self.valid_loss.values())
        valid_epochs, valid_acc = list(self.valid_loss.keys()), list(self.valid_acc.values())

        plt.scatter(train_epochs, train_values, label="traing loss") 
        plt.scatter(valid_epochs, valid_loss, label="valid loss")
        plt.scatter(valid_epochs, valid_acc, label="valid acc")
        plt.grid()
        plt.legend() 



class DiffussionTrainer(Trainer):
    def __init__(self, max_epochs):
        super().__init__(max_epochs=max_epochs)

    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
       
    def fit(self, data, model, diffusion):
        self.prepare_data(data)
        self.prepare_model(model=model)
        self.diffusion = diffusion

        init_batch = next(iter(self.train_dataloader))

        # print(init_batch[0].shape)
        # init_batch = next(iter(data.get_test_dataloader()))
        # init_batch = self.prepare_batch(init_batch)
        # self.model.init_network(init_batch[0])
        
        self.optim = self.model.configure_optimizer()

        self.training_loss = {}
        

        for self.epoch in range(self.max_epochs):
            print(f"currently running {self.epoch+1}")
            self.fit_epoch()
    
    def fit_epoch(self):
        self.model.train()
        train_loss = []
        for step, batch in enumerate(self.train_dataloader):
            self.optim.zero_grad()

            
            batch = self.prepare_batch(batch=batch)
            images = batch[0]
            # print(images.shape)
            
            t = self.diffusion.sample_timestamps(images.shape[0]).to(self.diffusion.device)
            x_t, noise = self.diffusion.noise_image(images, t)
            predicted_noise = self.model(x_t, t)
            l = self.model.loss(noise, predicted_noise)
            l.backward()
            self.optim.step()
            to_save = l.detach().cpu().item()
            train_loss.append(to_save)

        self.training_loss[self.epoch] = torch.mean(torch.tensor(train_loss))

        print("Training loop done 1")


        sampled_images = self.diffusion.sample(self.model, n=images.shape[0])
        print(sampled_images)
        self.save_images(sampled_images, os.path.join("results", f"epoch_{self.epoch}.jpg"))


    def save_images(self, images, path, **kwargs):    
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(path)



    def prepare_batch(self, batch):
        X, y = batch[0], batch[1]
        X = X.type(torch.float32)
        X = X.to(self.gpus[0])
        y = y.to(self.gpus[0])
        return [X, y]