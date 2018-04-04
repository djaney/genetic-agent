#!/usr/bin/python3
from evolution import Species
import argparse
import socket

parser = argparse.ArgumentParser(description='Genetic algorithm agent')
parser.add_argument('model_yaml_path',
                    help='path to yaml containing model information')

def main(args):
	with open(args.model_yaml_path) as f:
		yaml_string = f.read()

	agent = Species(model_yaml=yaml_string)


	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.bind(('', 8888))
	print ('Listening on port 8888')


	while True:
		try:
			# 1024 is the read buffer. larget than that we need to have multiple batch
			raw, addr = s.recvfrom(1024)
			data = raw.decode("utf-8").split(' ')
			cmd = data[0]
			idx = int(data[1])
			inp = data[2]
			res = None

			if 'act' == cmd:
				inp = inp.split(',')
				inp = [float(i) for i in inp]
				res = agent.act(inp,idx)
				res = res.tolist()
				res = [str(i) for i in res]
				res = ','.join(res)
			elif 'rec' == cmd:
				agent.record(inp,idx)
				if agent.is_ready_to_evolve():
					agent.evolve()
					res = '1'
				else:
					res = '0'
		except Exception as e:
			res = "error {0}".format(repr(e))

		if res != None:
			s.sendto(res.encode(), addr)



main(parser.parse_args())