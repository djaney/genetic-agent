#!/usr/bin/python3
from evolution import Species
import argparse

import tornado.ioloop
import tornado.web
import json

parser = argparse.ArgumentParser(description='Genetic algorithm agent')
parser.add_argument('input', type=int)
parser.add_argument('output', type=int)
parser.add_argument('hidden', type=int)
parser.add_argument('depth', type=int)
parser.add_argument('--species', default=1)
parser.add_argument('--population', default=10)
parser.add_argument('--port', default=8888)

agents = []


class ActHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        res = []
        for d in data:
            action = agents[d[0]].act(d[2], d[1])
            res.append([d[0], d[1], action])
        self.write(json.dumps(res))


class RecHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def post(self):
        print(self.request.body)

        # data = json.loads(self.request.body.encode('utf-8'))
        self.write("Hello, world")


def make_app():
    return tornado.web.Application([
        (r"/act", ActHandler),
        (r"/rec", RecHandler),
    ])


def main(args):
    for _ in range(args.species):
        agents.append(Species(args.input, args.output, args.hidden, args.depth, strain_count=args.population))

    print('listening {}'.format(args.port))
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()



main(parser.parse_args())
