from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from simple_network import Simple_Consistency_Regularisation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from DataGenerator import datagen, Get_Data




(xl_train, yl_train, xu_train), (x_test, y_test), (x_val, y_val) = Get_Data('CIFAR10', 0.1, 4000)
datagen.fit(xl_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=xl_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))


network = Simple_Consistency_Regularisation(model, datagen=datagen)
network.train(model, xl_train, yl_train, xu_train, x_val, y_val)