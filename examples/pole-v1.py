import gym

import numpy as np
from neat import Population, Printer

env = gym.make('CartPole-v1')

done = False

strain_count = 10
passing_score = 500
p = Population(1000, env.observation_space.shape[0], env.action_space.n)

max_reward = 0
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
                # env.render()
                if done:
                    break
            p.set_score(s, i, reward_sum)
            max_reward = np.max([reward_sum, max_reward])
            print(reward_sum)

    if max_reward > 100:
        break

    p.evolve()

pr = Printer(p.population[next(iter(p.population))][0])
pr.print()
