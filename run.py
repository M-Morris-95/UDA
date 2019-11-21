from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Trainer import Network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization

datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

#sys.stdout = open('output.txt','wt')

# Fetch and format the mnist data
(x_train, y_train), (x_test, y_test), = tf.keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255
datagen.fit(x_test.reshape((x_test.shape[0], 28, 28, 1)))

datset_size = np.size(y_train)
split = 0.05
split = int(np.floor(datset_size*split))

Lx = x_train[:split]
Ly = y_train[:split]

Ux = x_train[split:]

Ux = Ux[:,:,:,np.newaxis]
Lx = Lx[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]


# model = Sequential()
# # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
model = Sequential()
model.add(Conv2D(64, kernel_size=4, input_shape=(28, 28, 1), padding="same", strides=2, kernel_initializer=weight_init))

model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=2, strides=1))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=2, strides=1))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=2, strides=1))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, kernel_initializer=weight_init))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



network = Network(model, datagen)

network.train(train_x=Lx,
              train_y=Ly,
              unlabelled_x=Ux,
              val_x = x_test, val_y=y_test,
              epochs=20,
              Lambda = 0.5,
              labelled_batch_size=32,
              unlabelled_batch_size = 256)


network.model.save('UDA_Model.hdf5')
