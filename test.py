import gym

import numpy as np
from evolution import Species


env = gym.make('CartPole-v1')

done = False

agent = Species(input_count=4, output_count=2, hidden=1, depth=1, strain_count=100)


# learn
gen = 0
while True:

    scores = []
    max_score = 0
    print('generation {}'.format(agent.current_generation))
    for i in range(100):
        reward_sum = 0
        ob = env.reset()
        while True:
            action = agent.act(ob, i)
            ob, reward, done, info = env.step(np.argmax(action))
            reward_sum = reward_sum + reward
            scores.append(reward_sum)
            # env.render()
            if done:
                break
        agent.record(reward_sum, i)
        print('max score {}'.format(np.max(scores)))

    agent.evolve()
