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
            noise = torch.normal(means=torch.zeros_like(param), std=self.sigma)
            param.add(noise)
        return perturbed_model

    def reset(self):
        self.sigma = self.initial_sigma
        




