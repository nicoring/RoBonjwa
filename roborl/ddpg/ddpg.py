import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from random_process import OrnsteinUhlenbeckProcess
from memory import ReplayMemory


class DDPG:
    def __init__(self, env, actor_model, critic_model, memory, batch_size, gamma, 
                 tau, actor_lr, critic_lr, critic_decay, ou_theta=0.15, ou_sigma=0.2, render=None):
        self.env = env
        self.actor = actor_model
        self.actor_target = actor_model.clone()
        self.critic = critic_model
        self.critic_target = critic_model.clone()
        self.memory = ReplayMemory(memory)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.random_process = OrnsteinUhlenbeckProcess(env.action_space.shape[0],
                                                       theta=ou_theta, sigma=ou_sigma)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=critic_lr,
                                       weight_decay=critic_decay)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.render = render

    def update(self, target, source):
        zipped = zip(target.parameters(), source.parameters())
        for target_param, source_param in zipped:
            updated_param = target_param.data * (1 - self.tau) + \
                source_param.data * self.tau
            target_param.data.copy_(updated_param)

    def train_models(self):
        if len(self.memory) < self.batch_size:
            return None, None
        mini_batch = self.memory.sample_batch(self.batch_size)
        critic_loss = self.train_critic(mini_batch)
        actor_loss = self.train_actor(mini_batch)
        self.update(self.actor_target, self.actor)
        self.update(self.critic_target, self.critic)
        return critic_loss.data[0], actor_loss.data[0]

    def mse(self, inputs, targets):
        return torch.mean((inputs - targets)**2)

    def train_critic(self, batch):
        # forward pass
        pred_actions = self.actor_target(batch.next_states)
        target_q = batch.rewards + self.critic_target([batch.next_states, pred_actions]) * self.gamma
        pred_q = self.critic([batch.states, batch.actions])
        # backward pass
        loss = self.mse(pred_q, target_q)
        self.optim_critic.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim_critic.step()
        return loss

    def train_actor(self, batch):        
        # forward pass
        pred_mu = self.actor(batch.states)
        pred_q = self.critic([batch.states, pred_mu])
        # backward pass
        loss = -pred_q.mean()
        self.optim_actor.zero_grad()
        loss.backward()
#         for param in self.actor.parameters():
#             param.grad.data.clamp_(-1, 1)
        self.optim_actor.step()
        return loss

    def prep_state(self, s):
        return Variable(torch.from_numpy(s).float().unsqueeze(0))

    def select_action(self, state, exploration=True):
        action = self.actor(state)
        if exploration:
            noise = self.random_process.sample()
            action = action + Variable(torch.from_numpy(noise).float())
        return action

    def train(self, num_episodes):
        running_reward = None
        reward_sums = []
        losses = []
        overall_step = 0

        for episode_number in range(num_episodes):
            done = False
            state = self.prep_state(self.env.reset())
            reward_sum = 0
            self.random_process.reset()

            while not done:
                overall_step += 1
                if self.render is not None and episode_number % self.render == 0:
                    self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.data.numpy()[0])
                next_state = self.prep_state(next_state)
                reward = torch.FloatTensor([reward])
                self.memory.add(state, action, reward, next_state)
                state = next_state
                reward_sum += reward[0]
                losses.append(self.train_models())
            if self.render is not None and episode_number % self.render == 0:
                self.env.render(close=True)
            reward_sums.append(reward_sum)
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('episode: {} steps: {} reward: {}'.format(episode_number, overall_step, running_reward))
        return reward_sums, losses

    def run(self, render=True):
        state = self.env.reset()
        done = False
        reward_sum = 0
        while not done:
            if render:
                self.env.render()
            action = self.select_action(self.prep_state(state),
                                        exploration=False)
            state, reward, done, _ = self.env.step(action.data[0])
            reward_sum += reward
        return reward_sum
