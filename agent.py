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


def get_json_body(handler):
    return json.loads(handler.request.body.decode('utf-8'))


class AgentHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json')


class ActHandler(AgentHandler):
    def post(self):
        data = get_json_body(self)
        res = []
        for d in data:
            action = agents[d[0]].act(d[2], d[1])
            res.append([d[0], d[1], action.tolist()])
        json_string = json.dumps(res)

        self.write(json_string)


class RecHandler(AgentHandler):
    def post(self):
        data = get_json_body(self)
        res = []
        for d in data:
            agent = agents[d[0]]
            agent.record(float(d[2]), d[1])
            is_evolved = agent.is_ready_to_evolve()
            if is_evolved:
                agent.evolve()
            res.append([d[0], d[1], {'evolved': is_evolved}])
        json_string = json.dumps(res)

        self.write(json_string)


def make_app():
    return tornado.web.Application([
        (r"/act", ActHandler),
        (r"/rec", RecHandler),
    ])


def main(args):
    for _ in range(args.species):
        agents.append(Species(args.input, args.output, args.hidden, args.depth, strain_count=args.population))

    agents[0].strains[0].summary()
    print('listening {}'.format(args.port))
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()


main(parser.parse_args())
