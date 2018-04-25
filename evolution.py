import random
import numpy as np
import math
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

    def __init__(self, strain_count=10, mutation_chance=0.01, model_factory=None):
        self.shapes = None
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

        self.strains = []

        # strain count -1 due to adding best score as part of the generation
        best_score, best_strain, parents = self.pooling(self.strain_count-1)

        self.strains.append(best_strain)

        # record best score
        self.best = best_score

        # add the best strain back
        # self.strains.append(best_strain)
        for (father, mother) in parents:
            self.strains.append(self.breed(father, mother))

        # reset next gen
        self.next_gen = []

    """
    Choose parents
    """

    def pooling(self, pair_count):


        # sort by fittest
        self.next_gen = sorted(self.next_gen, key=lambda item: item[0], reverse=True)

        # record best score
        best_score = self.next_gen[0][0]
        best_strain = self.next_gen[0][1]
        total_score = math.ceil(reduce(lambda s, item: s + item[0], self.next_gen, 0))

        strain_pool_idx = []
        for idx, obj in enumerate(self.next_gen):
            # consider only the top half rank
            virtual_score = (self.strain_count/2) - idx
            if virtual_score <= 0:
                break

            multiplier = math.floor(virtual_score) ** 2
            for _ in range(multiplier):
                strain_pool_idx.append(idx)

        random.shuffle(strain_pool_idx)
        parents = []
        for _ in range(pair_count):
            father_index = strain_pool_idx[0]

            # count the same as father
            count = strain_pool_idx.count(father_index)

            # remove the same as father
            strain_pool_idx = list(filter(lambda a: a != father_index, strain_pool_idx))
            mother_index = strain_pool_idx[0]

            # put them back
            strain_pool_idx = strain_pool_idx + [father_index for _ in range(count)]
            random.shuffle(strain_pool_idx)

            father = self.next_gen[father_index][1]
            mother = self.next_gen[mother_index][1]

            parents.append((father, mother))

        return best_score, best_strain, parents

    """
    Breed 2 strands to produce one child
    """

    def breed(self, father, mother):
        # get the shape of each layer for later restoration
        shapes = self.get_model_shapes(father)

        father = flatten_strain(father)
        mother = flatten_strain(mother)

        # chance to switch
        if random.random() > 0.5:
            tmp = father
            father = mother
            mother = tmp

        strain_length = len(father)
        splice_point = random.randrange(1, strain_length-1)
        new_strain = father[:splice_point] + mother[splice_point:]

        # a chance to mutate
        for m_idx in range(len(new_strain)):
            if random.random() < self.mutation_chance:
                new_strain[m_idx] = new_strain[m_idx] + random.uniform(-0.5, 0.5)

        # reshape the strain
        return restore_strain(new_strain, shapes)

    """
    get shapes of the model layers
    """

    def get_model_shapes(self, sample):
        if self.shapes is None:
            self.shapes = [l.shape for l in sample]
        return self.shapes
