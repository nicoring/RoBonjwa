import argparse

import gym

from roborl.util.models import Actor, SharedControllerActor, Critic

def run(args):
    env = gym.make(args.env)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    if 'shared' in args.actor:
        actor = SharedControllerActor.load(args.actor)
    else:
        actor = Actor.load(args.actor)
    print(actor.run(env, render=args.render))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--no-render', dest='render', action='store_false', default=True)
    args = parser.parse_args()
    run(args)