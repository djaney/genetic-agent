from keras.models import Sequential
from keras.layers import Dense


def create():
    model = Sequential()
    model.add(Dense(2, input_shape=[4], activation='relu'))
    return model
