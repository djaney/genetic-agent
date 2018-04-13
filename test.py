import socket
import numpy as np
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('', 8888))

while True:
    scores = []
    for i in range(10):
        inp = 1
        s.send('act {} {}'.format(i, inp).encode())
        res = s.recv(1024)
        res = abs(float(res.decode('utf-8')))
        score = 10 - res - inp+1
        s.send('rec {} {}'.format(i, score).encode())
        scores.append(score)
    print(np.mean(scores))
    if np.mean(scores) == 10:
        break