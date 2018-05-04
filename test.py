import gym
import numpy as np
from urllib import request
import json


environments = []
for _ in range(10):
    environments.append(gym.make('CartPole-v1'))

done_list = []
score_list = {}
# learn
gen = 0
while True:

    max_score = 0
    data = []
    reward_sum = 0
    for i, env in enumerate(environments):
        score_list[i] = 0
        ob = env.reset()
        data.append([0, i, ob.tolist()])
    data = json.dumps(data)
    req = request.Request('http://localhost:8888/act', data=data.encode('utf-8'))
    resp = request.urlopen(req)
    resp_data = resp.read().decode('utf-8')
    resp_data = json.loads(resp_data)

    for i, env in enumerate(environments):
        print(done_list)
        if i in done_list:
            continue
        action = resp_data[i][2]
        ob, reward, done, info = env.step(np.argmax(action))
        score_list[i] = score_list[i] + reward
        if done:
            done_list.append(i)
