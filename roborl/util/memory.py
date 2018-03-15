import os
from collections import namedtuple, deque
import random
import pickle

import numpy as np
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
Batch = namedtuple('Batch', ('states', 'actions', 'rewards', 'next_states', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def add(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample_batch(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*transitions))
        states = torch.cat(batch.states)
        actions = torch.cat(batch.actions)
        rewards = Variable(torch.cat(batch.rewards).unsqueeze(1))
        next_states = torch.cat(batch.next_states)
        done = Variable(torch.from_numpy(np.where(batch.done, 0., 1.)).float().unsqueeze(1))
        if use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            done = done.cuda()
        return Batch(states, actions, rewards, next_states, done)

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        filename = os.path.join(path, 'memory.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path):
        filename = os.path.join(path, 'memory.pkl')
        with open(filename, 'rb') as f:
            transitions = pickle.load(f)
            self.memory.extend(transitions)
