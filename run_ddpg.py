import argparse
import signal
import os
import json

import gym

from roborl.ddpg.ddpg import DDPG
from roborl.util.models import Actor, Critic, SharedControllerActor


def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    if args.continue_training:
        if args.actortype == 'shared':
            actor = SharedControllerActor.load(os.path.join(args.save_path, 'actor-shared'))
        else:
            actor = Actor.load(os.path.join(args.save_path, 'actor'))
        critic = Critic.load(os.path.join(args.save_path, 'critic'))
    else:
        if args.actortype == 'shared':
            if 'actor_conf' not in args:
                raise ValueError('actor-conf arg is required for shared actor')
            with open(args.actor_conf) as f:
                conf = json.load(f)
            actor = SharedControllerActor(n_states, conf['controller_conf'], conf['controller_list'],
                                          args.actor_hidden, args.batchnorm, args.layernorm)
        else:
            actor = Actor(n_states, n_actions, args.actor_hidden, args.batchnorm, args.layernorm)
        critic = Critic(n_states, n_actions, args.critic_hidden, args.batchnorm)

    ddpg = DDPG(env, actor, critic, args.replay_memory, args.batch_size, args.gamma,
                args.tau, args.lr_actor, args.lr_critic, args.decay_critic,
                render=args.render, save_path=args.save_path, save_every=args.save_every,
                num_trainings=args.num_trainings, exploration_type=args.exploration_type,
                train_every=args.train_every, evaluate_every=args.evaluate_every,
                run_name=args.run_name, num_evaluations=args.num_evaluations)
    signal.signal(signal.SIGUSR1, lambda a, b: ddpg.save(args.save_path))
    if args.continue_training:
        ddpg.load_state(args.save_path)
        ddpg.load_memory(args.save_path)
        ddpg.load_optim_dicts(args.save_path)
    if args.warmup and not args.continue_training:
        ddpg.warmup(10*args.batch_size)
    rewards, eval_rewards, losses = ddpg.train(args.steps)


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
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--num_trainings', type=int, default=1)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--evaluate_every', type=int, default=1000)
    parser.add_argument('--exploration_type', choices=['action', 'param'], default='action')
    parser.add_argument('--continue', default=False, dest='continue_training', action='store_true')
    parser.add_argument('--batchnorm', default=False, dest='batchnorm', action='store_true')
    parser.add_argument('--layernorm', default=False, dest='layernorm', action='store_true')
    parser.add_argument('--actortype', default='non-shared', choices=['non-shared', 'shared'])
    parser.add_argument('--actor-conf')
    parser.add_argument('--run-name', default='')
    parser.add_argument('--num_evaluations', type=int, default=1)
    args = parser.parse_args()
    run(args)
