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


class SharedControllerActor(MyModule):
    def __init__(self, n_states, controller_conf, controller_list, n_hidden):
        """
        constructs a policy network with locally connected controllers
        that can share weights

        Args:
            n_states: number of states that are the input to the policy
            controller_conf: dictionary with confifs for low-level controllers
            controller_list: list of controller names, if one name appears multiple times
                then these controllers share weights
            n_hidden: number of hidden units in the fully connected layer

        >> # example controller conf:
        >> controller_conf = {
            'leg': {
                'actions': 4,
                'hidden': 50
            }
            'arm': {
                'actions': 3,
                'hidden': 50
            }
        }

        >> # example controller list:
        >> controller_list = ['arm', 'arm', 'leg', 'leg']
        """
        super().__init__()
        self.args = (n_states, controller_conf, controller_list, n_hidden)
        self.lin1 = nn.Linear(n_states, n_hidden)
        self.controller_inputs, self.controller = self.create_controllers(controller_conf, controller_list, n_hidden)
        self.controller_list = controller_list
        self.init_weights()

    def create_controllers(self, controller_conf, controller_list, n_hidden):
        shared_controller = {}
        for name, conf in controller_conf.items():
            # TODO: create arbitrary subnet based on conf
            shared_controller[name] = nn.Linear(conf['hidden'] , conf['actions'])

        controller_inputs = []
        for name in controller_list:
            n_output = controller_conf[name]['hidden']
            controller_inputs.append(nn.Linear(n_hidden, n_output))

        return controller_inputs, shared_controller

    def init_weights(self):
        for l in [self.lin1, self.lin2, *self.controller_inputs, *self.controller.values()]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'actor')

    def forward(self, x):
        x = F.relu(self.lin1(x))
        outs = []
        for name, input_layer in zip(self.controller_list, self.controller_inputs):
            xc = F.relu(input_layer(x))
            outs.append(self.controller[name](xc))
        out = torch.cat(outs, 1)
        return F.tanh(out)


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
