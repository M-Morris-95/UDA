from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Losses import LLoss, ULoss, ULoss2, evaluate, aug_loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image

import pandas as pd
import numpy as np


aug = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

train = pd.read_csv('data/train.csv')

y_train = train.label.values
x_train = train.drop(['label'], axis = 1)

x_train = np.reshape(x_train.values, (-1,28,28))
x_train = x_train[:,:,:,np.newaxis]

x_test = pd.read_csv('data/test.csv')
x_test = np.reshape(x_test.values, (-1,28,28))
x_test = x_test[:,:,:,np.newaxis]

x_train = x_train/255
x_test = x_test/255

x_val = x_train[:4200]
y_val = y_train[:4200]

x_train = x_train[4200:]
y_train = y_train[4200:]

aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)

aug.fit(x_train)





weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

model = Sequential()
model.add(Conv2D(64, kernel_size=4, input_shape=(28, 28, 1), padding="same", strides=2,
                 kernel_initializer=weight_init))
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


loss_history = []
accuracy_history = []

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer= keras.optimizers.Adam(),
              metrics=['accuracy'])

model.load_weights("mnist_model.hdf5")

model.fit_generator(aug.flow(x_train, y_train,batch_size=32),
                        validation_data=(x_val, y_val),
                        epochs = 10
                    )

model.save('mnist_model.hdf5')

predictions = np.argmax(model.predict(x_test), axis = 1)
index = np.linspace(1,28000,28000)
index = index.astype(int)

submission = pd.DataFrame({'ImageId': index, 'Label': predictions})
submission.to_csv('data/my_submission.csv', index = False)