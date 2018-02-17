import torch
import torch.nn as nn
import torch.nn.functional as F


class Cloneable:
    def clone(self):
        assert self.args
        model_clone = self.__class__(*self.args)
        model_clone.load_state_dict(self.state_dict())
        model_clone.train(False)
        return model_clone


class Actor(nn.Module, Cloneable):
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

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.tanh(self.lin3(x))
        return x


class Critic(nn.Module, Cloneable):
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

    def forward(self, x):
        s, a = x
        states_hidden = F.relu(self.lin_states(s))
        x = F.relu(self.lin1(torch.cat([states_hidden, a], 1)))
        x = self.lin2(x)
        return x
