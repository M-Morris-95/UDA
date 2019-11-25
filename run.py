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
import time, os, fnmatch, shutil, sys

import parser

parser = parser.GetParser()
args = parser.parse_args()
type = args.dataset

if type == 'CIFAR10':
    (x_train, y_train), (x_test, y_test), = tf.keras.datasets.cifar10.load_data()
elif type == 'MNIST':
    (x_train, y_train), (x_test, y_test), = tf.keras.datasets.mnist.load_data()
num_classes=10
split = args.split


if np.ndim(x_train) == 3:
    x_train = x_train[:,:,:,np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

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

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.3,  # set range for random shear
        zoom_range=0.2,  # set range for random zoom
        channel_shift_range=0.2,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)



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



network = Network(model, datagen, type)

network.train(train_x=Lx,
              train_y=Ly,
              unlabelled_x=Ux,
              val_x=x_test,
              val_y=y_test,
              epochs=args.epochs,
              Lambda=args.Lambda,
              TSA = args.TSA,
              usup = args.usup,
              labelled_batch_size = args.n_batch,
              unlabelled_batch_size = args.n_batch)

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)

os.chdir('logging')
os.mkdir(timestamp)
os.chdir(timestamp)

text_file = open("parameters.txt", "w")
text_file.write(' '.join(sys.argv[1:]))
text_file.close()

network.history.to_csv (r'history.csv', index = None, header=True)
network.model.save('UDA_Model.hdf5')
