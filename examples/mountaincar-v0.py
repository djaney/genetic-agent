import gym

import numpy as np
from evolution import Species

env = gym.make('MountainCar-v0')

done = False

strain_count = 100
generation_check = 1
agent = Species(input_count=2, output_count=3, hidden=3, depth=0,
                strain_count=strain_count, final_activation='softmax')

agent.strains[0].summary()

# learn
gen = 0
while True:

    scores = []
    max_score = 0

    for i in range(strain_count):
        reward_sum = 200
        max_position = -1.2
        min_position = 0.6
        ob = env.reset()
        while True:
            action = agent.act(ob, i)
            ob, reward, done, info = env.step(np.argmax(action))
            max_position = np.max([max_position, ob[0]])
            min_position = np.min([min_position, ob[0]])
            reward_sum = reward_sum + reward
            # env.render()
            if done:
                break
        reward_sum = reward_sum + max_position + (abs(min_position)/2)
        agent.record(reward_sum, i)
        scores.append(reward_sum)
    print("generation {} max score {}".format(agent.current_generation, np.max(scores)))

    agent.evolve()

    if 0 == agent.current_generation % generation_check:
        ob = env.reset()
        while True:
            action = agent.act(ob, 0)
            ob, reward, done, info = env.step(np.argmax(action))
            env.render()
            if done:
                break



