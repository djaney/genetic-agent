from keras.models import Sequential, Model
from keras.layers import Dense, Input

def create():
    a = Input(shape=(4,))
    b = Dense(2)(a)
    return Model(inputs=a, outputs=b)