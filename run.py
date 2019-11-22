from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Trainer import Network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization


import parser

parser = parser.GetParser()
args = parser.parse_args()




(x_train, y_train), (x_test, y_test), = tf.keras.datasets.cifar10.load_data()
num_classes=10
split = args.split

y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
x_train=x_train/np.max(x_train)
x_test=x_test/np.max(x_test)



datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)



datagen.fit(x_train)

Lx = x_train[:split]
Ly = y_train[:split]
Ux = x_train[split:]

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
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
model.add(Dense(num_classes))
model.add(Activation('softmax'))



network = Network(model, datagen)

network.train(train_x=Lx,
              train_y=Ly,
              unlabelled_x=Ux,
              val_x=x_test,
              val_y=y_test,
              epochs=args.epochs,
              Lambda=args.Lambda,
              TSA = args.TSA,
              usup = args.usup,
              labelled_batch_size=32,
              unlabelled_batch_size = 256)



network.model.save('UDA_Model.hdf5')
