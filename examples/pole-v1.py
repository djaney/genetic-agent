import gym

import numpy as np
from evolution import Species

env = gym.make('CartPole-v1')

done = False

strain_count = 10
passing_score = 500
agent = Species(input_count=env.observation_space.shape[0], output_count=env.action_space.n, hidden=1, depth=1,
                strain_count=strain_count)

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
            ob, reward, done, info = env.step(np.argmax(action))
            reward_sum = reward_sum + reward
            # env.render()
            if done:
                break
        agent.record(reward_sum, i)
        scores.append(reward_sum)
    print("generation {} max score {}".format(agent.current_generation, np.max(scores)))

    if np.max(scores) >= passing_score:
        break
    else:
        agent.evolve()

ob = env.reset()
while True:
    action = agent.act(ob, 0)
    ob, reward, done, info = env.step(np.argmax(action))
    scores.append(reward_sum)
    env.render()
    if done:
        ob = env.reset()

