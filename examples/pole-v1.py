import gym

import numpy as np
from neat import Population, Printer

env = gym.make('CartPole-v1')

done = False

strain_count = 10
passing_score = 500
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
    print(p.generation, max_reward, p.population.keys())
    if max_reward >= target_reward:
        break
    p.evolve()

print('Species {} is the winner'.format(winner[0]))

ob = env.reset()

while True:
    action = action = p.run(winner[0], winner[1], ob)
    ob, reward, done, info = env.step(np.argmax(action))
    env.render()
    if done:
        break

Printer(p.population[winner[0]][winner[1]]).print()

