import os
import pickle

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
        with open('%s-args.pkl' % filename, 'rb') as f:
            args = pickle.load(f)
        model = cls(*args)
        dict_filename = '%s.model' % filename
        model.load_state_dict(torch.load(dict_filename, map_location=lambda storage, loc: storage))
        return model

    def save(self, path, filename):
        args_file = os.path.join(path, '%s-args.pkl' % filename)
        with open(args_file, 'wb') as f:
            pickle.dump(self.args, f)
        torch.save(self.state_dict(), os.path.join(path, '%s.model' % filename))


class Actor(MyModule):
    def __init__(self, n_states, n_actions, n_hidden, use_batch_norm=False, use_layer_norm=False):
        super().__init__()
        self.args = (n_states, n_actions, n_hidden, use_batch_norm, use_layer_norm)
        if use_batch_norm and use_layer_norm:
            raise ValueError("Dont use both batch and layer norm")
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.lin1 = nn.Linear(n_states, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_actions)
        if self.use_batch_norm:
            self.bn_1 = nn.BatchNorm1d(n_hidden)
            self.bn_2 = nn.BatchNorm1d(n_hidden)
        if self.use_layer_norm:
            self.ln_1 = nn.LayerNorm(n_hidden)
            self.ln_2 = nn.LayerNorm(n_hidden)
        self.init_weights()

    def init_weights(self):
        for l in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'actor')

    def forward(self, x):
        x = self.lin1(x)
        if self.use_batch_norm:
            x = self.bn_1(x)
        if self.use_layer_norm:
            x = self.ln_1(x)
        x = F.relu(x)
        x = self.lin2(x)
        if self.use_batch_norm:
            x = self.bn_2(x)
        if self.use_layer_norm:
            x = self.ln_2(x)
        x = F.relu(x)
        x = F.tanh(self.lin3(x))
        return x


class SharedControllerActor(MyModule):
    def __init__(self, n_states, controller_conf, controller_list, n_hidden, use_batch_norm=False):
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
        self.args = (n_states, controller_conf, controller_list, n_hidden, use_batch_norm)
        self.use_batch_norm = use_batch_norm
        self.lin1 = nn.Linear(n_states, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.controller_inputs, self.controller = self.create_controllers(controller_conf, controller_list, n_hidden)
        self.controller_list = controller_list
        if use_batch_norm:
            self.bn_1 = nn.BatchNorm1d(n_hidden)
            self.bn_2 = nn.BatchNorm1d(n_hidden)
            self.controller_input_bns = self.controller_bn(self.controller_inputs)
        self.init_weights()

    def create_controllers(self, controller_conf, controller_list, n_hidden):
        shared_controller = {}
        for name, conf in controller_conf.items():
            # TODO: create arbitrary subnet based on conf
            l = nn.Linear(conf['hidden'] , conf['actions'])
            self.add_module(name, l)
            shared_controller[name] = l

        controller_inputs = []
        for i, name in enumerate(controller_list):
            n_output = controller_conf[name]['hidden']
            l = nn.Linear(n_hidden, n_output)
            self.add_module('controller_input_%d' % i, l)
            controller_inputs.append(l)

        return controller_inputs, shared_controller

    def controller_bn(self, controller_inputs):
        controller_input_bns = []
        for i, input_layer in enumerate(controller_inputs):
            bn = nn.BatchNorm1d(input_layer.out_features)
            self.add_module('controller_input_bn_%d' % i, bn)
            controller_input_bns.append(bn)
        return controller_input_bns

    def init_weights(self):
        for l in [self.lin1, self.lin2, *self.controller_inputs, *self.controller.values()]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'actor-shared')

    def forward(self, x):
        x = self.lin1(x)
        if self.use_batch_norm:
            x = self.bn_1(x)
        x = F.relu(x)
        x = self.lin2(x)
        if self.use_batch_norm:
            x = self.bn_2(x)
        x = F.relu(x)
 
        outs = []
        i = 0
        for name, input_layer in zip(self.controller_list, self.controller_inputs):
            xc = input_layer(x)
            if self.use_batch_norm:
                xc = self.controller_input_bns[i](xc)
                i += 1
            sc = F.relu(xc)
            outs.append(self.controller[name](xc))
        out = torch.cat(outs, 1)
        return F.tanh(out)


class Critic(MyModule):
    def __init__(self, n_states, n_actions, n_hidden, use_batch_norm=False):
        super().__init__()
        self.args = (n_states, n_actions, n_hidden, use_batch_norm)
        self.use_batch_norm = use_batch_norm
        self.lin_states = nn.Linear(n_states, n_hidden)
        self.lin1 = nn.Linear(n_hidden + n_actions, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        if self.use_batch_norm:
            self.bn_states = nn.BatchNorm1d(n_hidden)
            self.bn_1 = nn.BatchNorm1d(n_hidden)
        self.init_weights()

    def init_weights(self):
        for l in [self.lin_states, self.lin1, self.lin2]:
            nn.init.xavier_uniform(l.weight)

    def save(self, path):
        super().save(path, 'critic')

    def forward(self, x):
        s, a = x
        s = self.lin_states(s)
        if self.use_batch_norm:
            s = self.bn_states(s)
        states_hidden = F.relu(s)
        x = self.lin1(torch.cat([states_hidden, a], 1))
        if self.use_batch_norm:
            x = self.bn_1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
