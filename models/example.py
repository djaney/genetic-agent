from keras.models import Sequential, Model
from keras.layers import Dense, Input

def create():
    a = Input(shape=(1,))
    b = Dense(1)(a)
    return Model(inputs=a, outputs=b)