import argparse

import gym
import roboschool

from ddpg import DDPG
from model import Actor, Critic


def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(n_states, n_actions, args.actor_hidden)
    critic = Critic(n_states, n_actions, args.critic_hidden)
    try:
        # memory, batch_size, gamma, tau, actor_lr, critic_lr, critic_decay, 
        ddpg = DDPG(env, actor, critic, args.replay_memory, args.batch_size, args.gamma,
                    args.tau, args.lr_actor, args.lr_critic, args.decay_critic)
        rewards, losses = ddpg.train(args.episodes)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDPG implementation')
    parser.add_argument('--env', help='OpenAI Gym env name', default='RoboschoolInvertedPendulum-v1')
    parser.add_argument('--critic_hidden', type=int, default=100)
    parser.add_argument('--actor_hidden', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--replay_memory', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-4)
    parser.add_argument('--decay_critic', type=float, default=1e-2)

    args = parser.parse_args()
    run(args)
