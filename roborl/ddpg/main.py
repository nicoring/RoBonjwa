import argparse
import signal
import os

import gym


from ddpg import DDPG
from models import Actor, Critic

def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    if args.continue_training:
        actor = Actor.load(os.path.join(args.save_path, 'actor'))
        critic = Critic.load(os.path.join(args.save_path, 'critic'))
    else:
        actor = Actor(n_states, n_actions, args.actor_hidden, args.batchnorm, args.layernorm)
        critic = Critic(n_states, n_actions, args.critic_hidden, args.batchnorm)
    try:
        ddpg = DDPG(env, actor, critic, args.replay_memory, args.batch_size, args.gamma,
                    args.tau, args.lr_actor, args.lr_critic, args.decay_critic,
                    render=args.render, evaluate=args.evaluate, save_path=args.save_path,
                    save_every=args.save_every, train_per_step=args.train_per_step,
                    exploration_type=args.exploration_type)
        signal.signal(signal.SIGUSR1, lambda a, b: ddpg.save(args.save_path))
        if args.continue_training:
            ddpg.load_state(args.save_path)
            ddpg.load_memory(args.save_path)
            ddpg.load_optim_dicts(args.save_path)
        if args.warmup and not args.continue_training:
            ddpg.warmup(10*args.batch_size)
        rewards, losses = ddpg.train(args.steps)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDPG implementation')
    parser.add_argument('--env', help='OpenAI Gym env name', default='RoboschoolAnt-v1')
    parser.add_argument('--critic_hidden', type=int, default=100)
    parser.add_argument('--actor_hidden', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--replay_memory', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--decay_critic', type=float, default=1e-2)
    parser.add_argument('--render', default=False, dest='render', action='store_true')
    parser.add_argument('--warmup', default=False, dest='warmup', action='store_true')
    parser.add_argument('--evaluate', type=int, default=10)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--exploration_type', choices=['action', 'param'], default='action')
    train_when_parser = parser.add_mutually_exclusive_group(required=False)
    train_when_parser.add_argument('--train_per_episode', dest='train_per_step', action='store_false')
    train_when_parser.add_argument('--train_per_step', dest='train_per_step', action='store_true')
    parser.set_defaults(train_per_step=True)
    parser.add_argument('--batchnorm', default=False, dest='batchnorm', action='store_true')
    parser.add_argument('--continue', default=False, dest='continue_training', action='store_true')
    parser.add_argument('--layernorm', default=False, dest='layernorm', action='store_true')
    args = parser.parse_args()
    run(args)
