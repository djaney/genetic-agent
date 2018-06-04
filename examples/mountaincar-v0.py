
import gym

import numpy as np
from neat import Population, Printer

env = gym.make('MountainCar-v0')

done = False

strain_count = 10
passing_score = 500
p = Population(1000, 2, 3)
target_reward = 0.6
max_reward = -999999
winner = None
max_position = -1.2
min_position = 0.6
while True:
    status = p.get_status()
    for s in status.keys():
        output = []
        for i in range(status.get(s, 0)):
            ob = env.reset()
            reward_sum = 200
            while True:
                action = p.run(s, i, ob)
                ob, reward, done, info = env.step(np.argmax(action))
                max_position = np.max([max_position, ob[0]])
                min_position = np.min([min_position, ob[0]])
                reward_sum = reward_sum + reward
                if done:
                    break

            reward_sum = reward_sum + ((max_position + 1.2) - (min_position + 1.2))
            max_reward = np.max([reward_sum, max_reward])
            p.set_score(s, i, reward_sum)

            if max_position >= target_reward:
                winner = (s, i)
                break

        if max_position >= target_reward:
            break
    print('Generation: {} Score: {} Max position: {} Population: {}'.format(p.generation, reward_sum, max_position, p.population.keys()))
    if max_position >= target_reward:
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