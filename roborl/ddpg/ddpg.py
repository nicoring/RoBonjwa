import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter


from roborl.util.memory import ReplayMemory
from roborl.util.exploration import ActionNoiseExploration, ParamNoiseExploration


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor
writer = SummaryWriter()

class DDPG:
    def __init__(self, env, actor_model, critic_model, memory=10000, batch_size=64, gamma=0.99, 
                 tau=0.001, actor_lr=1e-4, critic_lr=1e-3, critic_decay=1e-2, ou_theta=0.15,
                 ou_sigma=0.2, render=None, save_path=None, save_every=10, render_every=10,
                 num_trainings=100, exploration_type='action', param_noise_bs=32, train_every=1,
                 evaluate_every=2000):
        self.env = env
        self.actor = actor_model
        self.actor_target = actor_model.clone()
        self.critic = critic_model
        self.critic_target = critic_model.clone()
        if use_cuda:
            for net in [self.actor, self.actor_target, self.critic, self.critic_target]:
                net.cuda()
        self.memory = ReplayMemory(memory)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        if exploration_type == 'action':
            self.exploration = ActionNoiseExploration(self.actor, env, ou_theta, ou_sigma)
        else:
            self.exploration = ParamNoiseExploration(self.actor, param_noise_bs, self.memory)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=critic_lr,
                                       weight_decay=critic_decay)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.render = render
        self.render_every = render_every
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.save_every = save_every
        self.num_trainings = num_trainings
        self.train_every = train_every
        self.evaluate_every = evaluate_every
        # state
        self.overall_step = 0
        self.overall_episode_number = 0
        self.running_reward = None
        self.reward_sums = []
        self.eval_reward_sums = []
        self.losses = []

    def add_graphs(self, actor, critic):
        sample_state = self.prep_state(np.random.uniform(size=self.env.observation_space.shape))
        sample_action = self.prep_state(np.random.uniform(size=self.env.action_space.shape))
        critic_input = [sample_state, sample_action]
        writer.add_graph(self.actor, sample_state)
        writer.add_graph(self.critic, critic_input)

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
        actor_loss = self.train_actor(mini_batch)
        critic_loss = self.train_critic(mini_batch)
        self.update(self.actor_target, self.actor)
        self.update(self.critic_target, self.critic)
        return actor_loss.data[0], critic_loss.data[0]

    def mse(self, inputs, targets):
        return torch.mean((inputs - targets)**2)

    def train_critic(self, batch):
        # forward pass
        pred_actions = self.actor_target(batch.next_states)
        target_q = batch.rewards + batch.done * self.critic_target([batch.next_states, pred_actions]) * self.gamma
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
        self.optim_actor.step()
        return loss

    def prep_state(self, s):
        return Variable(torch.from_numpy(s).float().unsqueeze(0))


    def step(self, action):
        next_state, reward, done, _ = self.env.step(action.data.cpu().numpy()[0])
        next_state = self.prep_state(next_state)
        reward = FloatTensor([reward])
        return next_state, reward, done

    def warmup(self, num_steps):
        warmup_step = 0
        while warmup_step <= num_steps:
            done = False
            state = self.prep_state(self.env.reset())
            self.exploration.reset()
            while not done:
                warmup_step += 1
                action = self.exploration.select_action(state)
                next_state, reward, done = self.step(action)
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

    def train(self, num_steps):
        simulation_step = 0 
        train_step = 0
        num_evaluations = 0

        while num_steps == None or simulation_step <= num_steps:
            self.overall_episode_number += 1
            done = False
            state = self.prep_state(self.env.reset())
            reward_sum = 0
            self.exploration.reset()
            while not done:
                self.overall_step += 1
                simulation_step += 1
                action = self.exploration.select_action(state)
                next_state, reward, done = self.step(action)
                self.memory.add(state, action, reward, next_state, done)
                state = next_state
                reward_sum += reward[0]
                if simulation_step % self.train_every == 0:
                    for _ in range(self.num_trainings):
                        train_step += 1
                        actor_loss, critic_loss = self.train_models()
                        self.losses.append([actor_loss, critic_loss])
                        writer.add_scalar('critic_loss', critic_loss, train_step)
                        writer.add_scalar('actor_loss', actor_loss, train_step)
                if simulation_step % self.evaluate_every == 0:
                    render_this_episode = self.render and (num_evaluations % self.render_every == 0)
                    evaluation_reward = self.actor.run(self.env, render=render_this_episode)
                    self.eval_reward_sums.append(evaluation_reward)
                    msg = '---------- eval_episode: {}  steps: {}  eval reward: {:.4f}'
                    print(msg.format(simulation_step // self.evaluate_every, self.overall_step, evaluation_reward))
                    writer.add_scalar('eval_reward', evaluation_reward, self.overall_step)
                    num_evaluations += 1
            self.reward_sums.append(reward_sum)
            self.running_reward = reward_sum if self.running_reward is None \
                                             else self.running_reward * 0.99 + reward_sum * 0.01
            writer.add_scalar('train_reward', reward_sum, self.overall_step)

            if self.save_path is not None and (self.overall_episode_number % self.save_every == 0):
                self.save(self.save_path)

            msg = 'episode: {}  steps: {}  running train reward: {:.4f}'
            print(msg.format(self.overall_episode_number, self.overall_step, self.running_reward))

        if self.save_path is not None:
            self.save(self.save_path)
        return self.reward_sums, self.eval_reward_sums, self.losses

    def run(self, render=True):
        state = self.env.reset()
        done = False
        reward_sum = 0
        while not done:
            if render:
                self.env.render()
            action = self.exploration.select_action(self.prep_state(state),
                                        exploration=False)
            state, reward, done, _ = self.env.step(action.data.cpu().numpy()[0])
            reward_sum += reward
        return reward_sum

    def load_state(self, path):
        filename = os.path.join(path, 'ddpg_state.pkl')
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)

    def load_memory(self, path):
        self.memory.load(path)

    def load_optim_dicts(self, path):
        critic_filename = os.path.join(path, 'critic.optim')
        actor_filename = os.path.join(path, 'actor.optim')
        self.optim_critic.load_state_dict(torch.load(critic_filename, map_location=lambda storage, loc: storage))
        self.optim_actor.load_state_dict(torch.load(actor_filename, map_location=lambda storage, loc: storage))

    def save(self, path):
        self.save_models(path)
        self.save_results(path, self.losses, self.reward_sums, self.eval_reward_sums)
        self.save_memory(path)
        self.save_optim_dicts(path)
        self.save_state(path)

    def save_optim_dicts(self, path):
        critic_filename = os.path.join(path, 'critic.optim')
        actor_filename = os.path.join(path, 'actor.optim')
        torch.save(self.optim_critic.state_dict(), critic_filename)
        torch.save(self.optim_actor.state_dict(), actor_filename)

    def save_state(self, path):
        params = ['overall_step', 'overall_episode_number', 'running_reward', 'reward_sums', 'losses']
        state = dict([(k, v) for k, v in self.__dict__.items() if k in params])
        filename = os.path.join(path, 'ddpg_state.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def save_models(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def save_results(self, path, losses, train_rewards, eval_rewards):
        losses = np.array([l for l in losses if l[0] is not None])
        train_rewards, eval_rewards = np.array(train_rewards), np.array(eval_rewards)
        np.savetxt(os.path.join(path, 'losses.csv'), losses, delimiter=',', header='critic,actor', comments='')
        np.savetxt(os.path.join(path, 'train_rewards.csv'), train_rewards, delimiter=',')
        np.savetxt(os.path.join(path, 'eval_rewards.csv'), eval_rewards, delimiter=',')

    def save_memory(self, path):
        self.memory.save(path)
