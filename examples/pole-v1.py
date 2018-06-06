import gym
import argparse
import numpy as np
import sys
from neat import Population, Printer

sys.setrecursionlimit(2000)


def play():
    env = gym.make('CartPole-v1')
    ob = env.reset()

    p = Population.load('cart-pole.pkl')

    winner = p.get_winner()

    while True:
        action = winner.run(ob)
        ob, reward, done, info = env.step(np.argmax(action))
        env.render()
        if done:
            break


def train():
    env = gym.make('CartPole-v1')

    p = Population(1000, env.observation_space.shape[0], env.action_space.n)
    target_reward = 500
    max_reward = 0
    winner = None
    while True:
        status = p.get_status()
        for s in status.keys():
            output = []
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
                if max_reward >= target_reward:
                    winner = (s, i)
                    break
            if max_reward >= target_reward:
                break

        try:
            p.save('cart-pole.pkl')
        except RuntimeError as e:
            print('error saving: {}'.format(str(e)))

        print(p.generation, max_reward, p.population.keys())
        if max_reward >= target_reward:
            break
        p.evolve()

    print('Species {} is the winner'.format(winner[0]))


def main(args):
    if args.command == "train":
        train()

    elif args.command == "play":
        play()


parser = argparse.ArgumentParser()

parser.add_argument("command")

main(parser.parse_args())
