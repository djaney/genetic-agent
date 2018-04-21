import socket
import numpy as np
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('', 8888))

while True:
    scores = []
    for i in range(10):
        inp = 1
        s.send('act 0 {} {}'.format(i, inp).encode())
        res = s.recv(1024)
        res = float(res.decode('utf-8'))

        score = 10 - abs(res - (inp+1))
        s.send('rec 0 {} {}'.format(i, score).encode())
        scores.append(score)
    print(np.max(scores))
    if np.max(scores) > 9.9:
        break

print('TEST')
s.send('act 0 0 1'.encode())
print(1, s.recv(1024))
