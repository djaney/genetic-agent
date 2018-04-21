#!/usr/bin/python3
from evolution import Species
import argparse
import socket

parser = argparse.ArgumentParser(description='Genetic algorithm agent')
parser.add_argument('model_factory', help='import path of model factory')
parser.add_argument('--species', default=1)
parser.add_argument('--population', default=10)
parser.add_argument('--port', default=8888)


def main(args):

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', args.port))

    agents = []
    for _ in range(args.species):
        agents.append(Species(model_factory=args.model_factory, strain_count=args.population))

    print('Listening on port {}'.format(args.port))

    while True:
        raw, addr = s.recvfrom(1024)
        try:
            # 1024 is the read buffer. larget than that we need to have multiple batch
            data = raw.decode("utf-8").split(' ')
            cmd = data[0]
            species_idx = int(data[1])
            idx = int(data[2])
            inp = data[3]
            res = None
            agent = agents[species_idx]

            if 'act' == cmd:
                inp = inp.split(',')
                inp = [float(i) for i in inp]
                res = agent.act(inp, idx)
                res = res.tolist()
                res = [str(i) for i in res]
                res = ','.join(res)
            elif 'rec' == cmd:
                inp = float(inp)
                agent.record(inp, idx)
                print('rec {} score {}, {} remaining'.format(idx, inp, len(agent.strains) - len(agent.next_gen)))
                if agent.is_ready_to_evolve():
                    agent.evolve()
                    print('evolve')
                    res = '1'
                else:
                    res = '0'
        except Exception as e:
            res = "error {0}".format(repr(e))

        if res is not None:
            s.sendto(res.encode(), addr)


main(parser.parse_args())
