import random
import numpy as np
import tensorflow as tf
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Input

a = Input(shape=(2,))
b = Dense(2)(a)
model = Model(inputs=a, outputs=b)
print(model.to_yaml())