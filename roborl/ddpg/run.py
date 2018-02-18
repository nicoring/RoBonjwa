import argparse

import gym
import roboschool

from ddpg import DDPG
from models import Actor, Critic

def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor.load(args.actor)
    critic = Critic.load(args.critic)
    try:
        ddpg = DDPG(env, actor, critic)
        print(ddpg.run(render=args.render))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--critic', required=True)
    parser.add_argument('--no-render', dest='render', action='store_false', default=True)
    args = parser.parse_args()
    run(args)