import random
import numpy as np
from operator import mul
from functools import reduce


def restore_strain(strain, shapes):
    new_strain = []
    start = 0
    for i, s in enumerate(shapes):
        end = start + reduce(mul, s)
        new_strain.append(np.array(strain[start:end]).reshape(shapes[i]))
        start = end
    return new_strain


def flatten_strain(strain):
    new_strain = []
    for l in strain:
        new_strain = new_strain + l.ravel().tolist()
    return new_strain


class Species(object):
    """
    Genetic evolution agent


    """

    def __init__(self, strain_count=10, mutation_chance=0.001, model_factory=None):
        self.strains = []
        self.next_gen = []
        self.strain_count = strain_count

        self.model_factory = model_factory

        self.model = self.create_model()
        self.mutation_chance = mutation_chance
        self.best = 0

        for _ in range(self.strain_count):
            self.strains.append(self.create_model().get_weights())

    def generation_size(self):
        return len(self.strains)

    def act(self, observation, strain_index):
        self.model.set_weights(self.strains[strain_index])
        y = self.model.predict(np.array([observation]), batch_size=1)
        return y[0]

    def record(self, reward, strain_index):
        hit = len([s for s in self.next_gen if s[2] == strain_index])
        if 0 == hit:
            self.next_gen.append((reward, self.strains[strain_index], strain_index))

    def create_model(self):
        if self.model_factory is None:
            raise Exception('No model factory')
        factory = __import__(self.model_factory, globals(), locals(), ['create'])
        return factory.create()

    def get_best_reward(self):
        return self.best

    def is_ready_to_evolve(self):
        return len(self.strains) == len(self.next_gen)

    def evolve(self):

        # find fittest
        self.next_gen = sorted(self.next_gen, key=lambda item: item[0], reverse=True)
        # breed
        father = self.next_gen[0][1]
        mother = self.next_gen[1][1]
        self.best = self.next_gen[0][0]

        # get the shape of each layer
        shapes = [l.shape for l in father]
        self.strains = []
        self.next_gen = []

        # add the best strain back into the pool
        self.strains.append(father)

        # convert to full numpy and get shape
        father = flatten_strain(father)
        mother = flatten_strain(mother)

        while len(self.strains) < self.strain_count:
            new_strain = []
            for i in range(len(father)):
                if 0 == random.randrange(0, 1):
                    new_strain.append(father[i])
                else:
                    new_strain.append(mother[i])

                # a chance to mutate
                if random.random() < self.mutation_chance:
                    m_idx = random.randrange(0, len(new_strain))
                    new_strain[m_idx] = random.uniform(-1, 1) + 0.000001
            # reshape the strain
            new_strain = restore_strain(new_strain, shapes)
            self.strains.append(new_strain)
