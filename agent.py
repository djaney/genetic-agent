from evolution import Species

with open('models/model_example.yaml') as f:
	yaml_string = f.read()

agent = Species(model_yaml=yaml_string)
# evolve in 10 generations
for _ in range(10):

	for i in range(10):
		agent.act([1,2,3,4],i) # observation, index

	for i in range(10):
		agent.record(1,i) # reward, index

	agent.evolve()
