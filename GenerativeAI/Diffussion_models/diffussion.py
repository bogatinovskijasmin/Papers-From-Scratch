from imports import *


class Diffussion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, size=64, device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = size

        self.beta = self.define_schedule().to(device=self.device)
        self.alfa = 1 - self.beta
        self.alfa_hat = torch.cumprod(self.alfa, dim=0)
    
    def define_schedule(self, ):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timestamps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    

    def noise_image(self, x, t):
        sqrt_alfa_hat = torch.sqrt(self.alfa_hat[t])[:, None, None, None]
        sqrt_one_minus_alfa_hat = torch.sqrt(1-self.alfa_hat[t])[:, None, None, None]
        eps = torch.rand_like(x)
        return sqrt_alfa_hat*x + sqrt_one_minus_alfa_hat*eps, eps
    


    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alfa[t][:, None, None, None]
                alpha_hat = self.alfa_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
            
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
        model.train()
        x = (x.clamp(-1, 1)+1)/2
        x = (x*255).type(torch.uint8)
        return x





