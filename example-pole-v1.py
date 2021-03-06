import gym
import argparse
import numpy as np
import sys
import os
from neat import Population, Printer

sys.setrecursionlimit(2000)
FILENAME = 'cart-pole.pkl'
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FULLNAME = '{}/{}'.format(DIR_PATH, 'save/cart-pole.pkl')


def play():
    env = gym.make('CartPole-v1')
    ob = env.reset()

    p = Population.load(FULLNAME)

    winner = p.get_winner()

    while True:
        action = winner.run(ob)
        ob, reward, done, info = env.step(np.argmax(action))
        env.render()
        if done:
            ob = env.reset()


def train():
    env = gym.make('CartPole-v1')

    try:
        p = Population.load(FULLNAME)
        print('Existing state loaded')
    except FileNotFoundError as e:
        print('Creating new state')
        p = Population(1000, env.observation_space.shape[0], env.action_space.n)

    max_reward = 0
    while True:
        try:

            status = p.get_status()
            for s in status.keys():
                for i in range(status.get(s, 0)):
                    ob = env.reset()
                    reward_sum = 0
                    while True:
                        action = p.run(s, i, ob)
                        ob, reward, done, info = env.step(np.argmax(action))
                        reward_sum = reward_sum + reward
                        if done:
                            break
                    p.set_score(s, i, reward_sum)
                    max_reward = np.max([reward_sum, max_reward])

            print(p.generation, max_reward, p.population.keys())
            p.evolve()

        except KeyboardInterrupt as e:

            try:
                print('\nsaving before exit')
                p.save(FULLNAME)
                sys.exit('Bye!')
            except RuntimeError as e:
                print('error saving: {}'.format(str(e)))


def main(args):
    if args.command == "train":
        train()

    elif args.command == "play":
        play()


parser = argparse.ArgumentParser()

parser.add_argument("command")
main(parser.parse_args())
