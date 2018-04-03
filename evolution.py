import random
import numpy as np
import tensorflow as tf
import numpy
from keras.models import Model, model_from_yaml
from keras.layers import Input, Dense
from operator import mul
from functools import reduce
class Species(object):
    """
    Genetic evolution agent


    """
    def __init__(self, strain_count = 10, mutation_chance=0.001, model_yaml= None):
        self.strains = []
        self.nextGen = []
        self.strain_count = strain_count

        if model_yaml == None:
            self.model = model_from_yaml(model_from_yaml(model_yaml))
        else:
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
        return y[0],self.strains[strain_index]

    def record(self, reward, strain_index):
        self.nextGen.append((reward,self.strains[strain_index]))

    def create_model(self):
        a = Input(shape=(4,))
        b = Dense(2)(a)
        return Model(inputs=a, outputs=b)

    def get_best_reward(self):
        return self.best

    def evolve(self):

        # find fittest
        self.nextGen = sorted(self.nextGen, key=lambda item: item[0], reverse=True)
        # breed
        father = self.nextGen[0][1]
        mother = self.nextGen[1][1]
        self.best = self.nextGen[0][0]

        # get the shape of each layer
        shapes = [l.shape for l in father]
        self.strains = []
        self.nextGen = []


        # add the best strain back into the pool
        self.strains.append(father)

    

        #convert to full numpy and get shape
        father = self.flatten_strain(father)
        mother = self.flatten_strain(mother)
        
        while len(self.strains) < self.strain_count:
            new_strain = []
            for i in range(len(father)):
                if 0 == random.randrange(0,1):
                    new_strain.append(father[i])
                else:
                    new_strain.append(mother[i])

                # a chance to mutate
                if random.random() < self.mutation_chance:
                    mIdx = random.randrange(0, len(new_strain))
                    new_strain[mIdx] = random.uniform(-1, 1)+0.000001
            # reshape the strain
            new_strain = self.restore_strain(new_strain, shapes)
            self.strains.append(new_strain)



    def flatten_strain(self, strain):
        new_strain = []
        for l in strain:
            new_strain = new_strain + l.ravel().tolist()
        return new_strain

    def restore_strain(self, strain, shapes):
        new_strain = []
        start = 0
        for i, s in enumerate(shapes):
            end = start + reduce(mul, s)
            new_strain.append(np.array(strain[start:end]).reshape(shapes[i]))
            start = end
        return new_strain