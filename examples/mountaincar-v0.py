import gym

import numpy as np
from evolution import Species

env = gym.make('MountainCar-v0')

done = False

strain_count = 10
generation_check = 10
agent = Species(input_count=2, output_count=3, hidden=3, depth=0,
                strain_count=strain_count, final_activation='softmax')

agent.strains[0].summary()


def play(a, e, index, render=False):
    reward_sum = 200
    max_position = -1.2
    min_position = 0.6
    ob = e.reset()
    while True:
        action = a.act(ob, index)
        ob, reward, done, info = e.step(np.argmax(action))
        max_position = np.max([max_position, ob[0]])
        min_position = np.min([min_position, ob[0]])
        reward_sum = reward_sum + reward
        if render:
            env.render()

        if done:
            break

    # negative_reward + distance covered
    reward_sum = reward_sum + ((max_position + 1.2) - (min_position + 1.2))

    return reward_sum


# learn
while True:

    scores = []
    max_score = 0

    # play every strain
    for i in range(strain_count):
        score = play(agent, env, i)
        agent.record(score, i)
        scores.append(score)
    print("generation {} max score {}".format(agent.current_generation, np.max(scores)))

    agent.evolve()

    if 0 == agent.current_generation % generation_check:
        play(agent, env, 0, True)
