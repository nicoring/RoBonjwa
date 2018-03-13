import numpy as np
import torch
from torch.autograd import Variable

from random_process import OrnsteinUhlenbeckProcess
from noise import ParamNoise


use_cuda = torch.cuda.is_available()

class ActionNoise:

    def __init__(self, actor, env, ou_theta, ou_sigma):
        self.actor = actor
        self.random_process = OrnsteinUhlenbeckProcess(env.action_space.shape[0],
                                                       theta=ou_theta, sigma=ou_sigma)
    
    def select_action(self, state, exploration=True):
        if use_cuda:
            state = state.cuda()
        self.actor.eval()
        action = self.actor(state)
        self.actor.train()
        if exploration:
            noise = Variable(torch.from_numpy(self.random_process.sample()).float())
            if use_cuda:
                noise = noise.cuda()
            action = action + noise
        return action
    
    def reset(self):
        self.random_process.reset()
    
class ParamNoise:

    def __init__(self, actor, batch_size, memory):
        self.actor = actor
        self.param_noise = ParamNoise(batch_size, memory)
        self.perturbed_model = self.param_noise.perturb_model(self.actor)

    def select_action(self, state, exploration=True):
        if use_cuda:
            state = state.cuda()
        if exploration:
            self.perturbed_model.eval()
            action = self.perturbed_model(state)
            self.perturbed_model.train()
            self.param_noise.update_sigma(self.actor, self.perturbed_model)
        else:
            self.actor.eval()
            action = self.actor(state)
            self.actor.train()
        return action
        
    def reset(self):
        self.param_noise.reset()
        self.perturbed_model = self.param_noise.perturb_model(self.actor)
        
        
        
