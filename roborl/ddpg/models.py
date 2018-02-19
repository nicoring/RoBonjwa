import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def clone(self):
        assert self.args
        model_clone = self.__class__(*self.args)
        model_clone.load_state_dict(self.state_dict())
        model_clone.train(False)
        return model_clone

    @classmethod
    def load(cls, filename):
        args = map(int, os.path.basename(filename).split('.')[0].split('-')[1].split('_'))
        model = cls(*args)
        model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        return model

    def save(self, path, filename):
        params_string = '_'.join(map(str, self.args))
        torch.save(self.state_dict(), os.path.join(path, '%s-%s.model' % (filename, params_string)))


class Actor(MyModule):
    def __init__(self, n_states, n_actions, n_hidden):
        super().__init__()
        self.args = (n_states, n_actions, n_hidden)
        self.lin1 = nn.Linear(n_states, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_actions)
        self.init_weights()

    def init_weights(self):
        for l in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'actor')

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.tanh(self.lin3(x))
        return x


class Critic(MyModule):
    def __init__(self, n_states, n_actions, n_hidden):
        super().__init__()
        self.args = (n_states, n_actions, n_hidden)
        self.lin_states = nn.Linear(n_states, n_hidden)
        self.lin1 = nn.Linear(n_hidden + n_actions, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.init_weights()

    def init_weights(self):
        for l in [self.lin_states, self.lin1, self.lin2]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'critic')

    def forward(self, x):
        s, a = x
        states_hidden = F.relu(self.lin_states(s))
        x = F.relu(self.lin1(torch.cat([states_hidden, a], 1)))
        x = self.lin2(x)
        return x
