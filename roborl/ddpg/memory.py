from collections import namedtuple, deque
import random

import torch
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))
Batch = namedtuple('Batch', ('states', 'actions', 'rewards', 'next_states'))


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
        return Batch(states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)
