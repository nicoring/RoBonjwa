import numpy as np
import torch

from torch.autograd import Variable

class ParamNoise:
    def __init__(self, batch_size, memory, sigma=0.1, delta=0.1, alpha=1.01):
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory = memory
        self.sigma = sigma
        self.delta = delta
        self.initial_sigma = sigma

    def distance(self, model, perturbed_model):
        ''' DDPG Distance '''
        batch = self.memory.sample_batch(self.batch_size)
        return torch.sqrt(torch.mean((model(batch.states) - perturbed_model(batch.states))**2))

    def update_sigma(self, model, perturbed_model):
        '''Updates sigma at each timestep'''
        if len(self.memory) < self.batch_size:
            return
        dist = self.distance(model, perturbed_model)
        if dist.data.numpy() <= self.delta:
            self.sigma *= self.alpha
        else:
            self.sigma *= 1 / self.alpha
    
    def perturb_model(self, model):
        perturbed_model = model.clone()
        for param in perturbed_model.parameters():
            noise = torch.normal(mean=torch.zeros_like(param), std=self.sigma)
            param.requires_grad = False
            param.add_(noise)
        return perturbed_model

    def reset(self):
        self.sigma = self.initial_sigma
        

class OrnsteinUhlenbeckProcess:
    def __init__(self, shape, theta=0.15, sigma=0.2, dt=1e-2):
        self.theta = theta
        self.mu = 0
        self.sigma = sigma
        self.dt = dt
        self.last = 0
        self.shape = np.atleast_1d(shape)

    def sample(self):
        self.last = self.last + self.theta * (self.mu - self.last) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.randn(*self.shape)
        return self.last

    def reset(self, sigma=None):
        if sigma is not None:
            self.sigma = sigma
        self.last = 0




