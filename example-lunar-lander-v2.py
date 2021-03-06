import gym
import argparse
import numpy as np
import sys
import os
from neat import Population, Printer

sys.setrecursionlimit(2000)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FULLNAME = '{}/{}'.format(DIR_PATH, 'save/lunar-lander.pkl')
AUTOSAVE = '{}/{}'.format(DIR_PATH, 'save/lunar-lander-autosave.pkl')


def play():
    env = gym.make('LunarLanderContinuous-v2')
    ob = env.reset()

    p = Population.load(FULLNAME)

    winner = p.get_winner()

    while True:
        action = action_final_activation(winner.run(ob))
        print('Action: {}'.format(action))
        ob, reward, done, info = env.step(action)
        env.render()
        if done:
            ob = env.reset()


def print_population():
    g = Population.load(FULLNAME).get_winner()
    Printer(g).print()


def action_final_activation(action):
    return np.array([np.clip(a, -1, 1) for a in action])


def train():
    env = gym.make('LunarLanderContinuous-v2')

    try:
        p = Population.load(FULLNAME)
        print('Existing state loaded')
    except FileNotFoundError as e:
        print(str(e) + '. Creating new state')
        p = Population(10000, env.observation_space.shape[0], env.action_space.shape[0])

    while True:
        try:
            max_reward = -99999
            status = p.get_status()
            for s in status.keys():
                for i in range(status.get(s, 0)):
                    ob = env.reset()
                    reward_sum = 0
                    while True:
                        action = action_final_activation(p.run(s, i, ob))
                        ob, reward, done, info = env.step(action)
                        reward_sum = reward_sum + reward
                        if done:
                            break
                    p.set_score(s, i, reward_sum)
                    max_reward = np.max([reward_sum, max_reward])
            print(p.generation, max_reward, p.population.keys())

            try:
                p.save(AUTOSAVE)
            except RuntimeError as e:
                print('error saving: {}'.format(str(e)))

            p.evolve()

        except KeyboardInterrupt as e:

            try:
                print('\nsaving before exit')
                p.save(FULLNAME)
                sys.exit('Bye!')
            except RuntimeError as e:
                print('error saving: {}'.format(str(e)))


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def main(args):
    if args.command == "train":
        train()

    elif args.command == "play":
        play()

    elif args.command == "print":
        print_population()


parser = argparse.ArgumentParser()

parser.add_argument("command")
main(parser.parse_args())
