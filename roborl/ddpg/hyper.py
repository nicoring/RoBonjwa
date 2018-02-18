import os
import argparse
import itertools as it
from copy import deepcopy

import gym
import roboschool

from ddpg import DDPG
from model import Actor, Critic

def all(args):
    for i in range(args.runs):
        learning_rate(args, i)
        tau(args, i)
        gamma(args, i)
        replay_memory(args, i)

def learning_rate(base_args, i):
    actor_lrs = [1e-3, 1e-4, 1e-5]
    critic_lrs = [1e-2, 1e-3, 1e-4]
    lr_pairs = zip(actor_lrs, critic_lrs)
    for a_lr, c_lr in lr_pairs:
        args = deepcopy(base_args)
        folder = str(a_lr) + '-' + str(c_lr)
        path = os.path.join(args.save_path, 'learningrates', folder, str(i))
        os.makedirs(path, exist_ok=True)
        args.lr_actor = a_lr
        args.lr_critic = c_lr
        args.save_path = path
        run(args)

def tau(base_args, i):
    for tau in [0.1, 0.01, 0.001, 0.0001]:
        args = deepcopy(base_args)
        folder = str(tau)
        path = os.path.join(args.save_path, 'taus', folder, str(i))
        os.makedirs(path, exist_ok=True)
        args.save_path = path
        args.tau = tau
        run(args)


def gamma(base_args, i):
    for gamma in [0.99, 0.999, 0.9999]:
        args = deepcopy(base_args)
        folder = str(gamma)
        path = os.path.join(args.save_path, 'gammas', folder, str(i))
        os.makedirs(path, exist_ok=True)
        args.save_path = path
        args.gamma = gamma
        run(args)


def replay_memory(base_args, i):
    for size in [1e+3, 1e+4, 1e+5]:
        size = int(size)
        args = deepcopy(base_args)
        folder = str(size)
        path = os.path.join(args.save_path, 'replay_sizes', folder, str(i))
        os.makedirs(path, exist_ok=True)
        args.save_path = path
        args.replay_memory = size
        run(args)

def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(n_states, n_actions, args.actor_hidden)
    critic = Critic(n_states, n_actions, args.critic_hidden)
    try:
        ddpg = DDPG(env, actor, critic, args.replay_memory, args.batch_size, args.gamma,
                    args.tau, args.lr_actor, args.lr_critic, args.decay_critic,
                    render=args.render, evaluate=args.evaluate, save_path=args.save_path,
                    save_every=args.save_every)
        rewards, losses = ddpg.train(args.steps)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDPG implementation')
    parser.add_argument('experiment')
    parser.add_argument('--env', help='OpenAI Gym env name', default='RoboschoolInvertedPendulum-v1')
    parser.add_argument('--critic_hidden', type=int, default=20)
    parser.add_argument('--actor_hidden', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--replay_memory', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-4)
    parser.add_argument('--decay_critic', type=float, default=1e-2)
    parser.add_argument('--render', default=False, dest='render', action='store_true')
    parser.add_argument('--evaluate', type=int, default=10)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)

    experiments = {
        'all': all,
        'learning_rate': learning_rate,
        'tau': tau,
        'gamma': gamma,
        'replay_memory': replay_memory
    }

    args = parser.parse_args()
    assert(args.experiment in experiments)
    experiments[args.experiment](args)
