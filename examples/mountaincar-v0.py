import gym

import numpy as np
from evolution import Species

env = gym.make('MountainCarContinuous-v0')

done = False

strain_count = 10
generation_check = 10
agent = Species(input_count=env.observation_space.shape[0], output_count=env.action_space.shape[0], hidden=2, depth=1,
                strain_count=strain_count, final_activation='tanh')

# learn
gen = 0
while True:

    scores = []
    max_score = 0

    for i in range(strain_count):
        reward_sum = 0
        ob = env.reset()
        while True:
            action = agent.act(ob, i)
            ob, reward, done, info = env.step(action)
            reward_sum = np.max([reward_sum, reward])
            # env.render()
            if done:
                break
        agent.record(reward_sum, i)
        scores.append(reward_sum)
    print("generation {} max score {}".format(agent.current_generation, np.max(scores)))

    if 0 == agent.current_generation % generation_check:
        agent.evolve()
        ob = env.reset()
        while True:
            action = agent.act(ob, 0)
            ob, reward, done, info = env.step(action)
            scores.append(reward_sum)
            env.render()
            if done:
                break
    else:
        agent.evolve()



