import random
import numpy as np
import tensorflow as tf
import numpy
from keras.models import Model
from keras.layers import Input, Dense
class Species(object):
    """
    Genetic evolution agent


    """
    def __init__(self, strain_count = 10, mutation_chance=0.001):
        self.strains = []
        self.nextGen = []
        self.strain_count = strain_count
        self.model = self.create_model()
        self.weights_shape = self.model.get_weights().shape
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

    def record_result(self, reward, strain_index):
        self.nextGen.append((reward,self.strains[strain_index].flatten()))

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

        self.strains = []
        self.nextGen = []

        # add the best strain back into the pool
        self.strains.append([numpy.reshape(father, self.weights_shape)])

        for _ in range(self.strain_count):
            newStrain = []
            for i in range(len(father)):
                if 0 == random.randrange(0,1):
                    newStrain.append(father[i])
                else:
                    newStrain.append(mother[i])

                # a chance to mutate
                if random.random() < self.mutation_chance:
                    mIdx = random.randrange(0, len(newStrain))
                    newStrain[mIdx] = random.uniform(-1, 1)+0.000001

            newStrain = numpy.reshape(newStrain, (4,2))
            self.strains.append([newStrain])



