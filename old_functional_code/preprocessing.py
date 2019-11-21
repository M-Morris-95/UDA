from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Losses import LLoss, ULoss, ULoss2, evaluate, aug_loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from progressbar import ProgressBar
pbar = ProgressBar()
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

Ly = train.label.values
Lx = train.drop(['label'], axis = 1)

Lx = np.reshape(Lx.values, (-1,28,28))
Lx = Lx[:,:,:,np.newaxis]

test = pd.read_csv('test.csv')
Ux = np.reshape(test.values, (-1,28,28))
Ux = Ux[:,:,:,np.newaxis]

Lx = Lx/255
Ux = Ux/255


datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)


datagen.fit(Ux.reshape((Ux.shape[0], 28, 28, 1)))

split = 0.01
datset_size = 60000
size = np.floor(datset_size*split)
size = int(size)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop()
loss_history = []
accuracy_history = []

def train_step(x,  model, Lambda = 1, sup = True, labels = 0):
    with tf.GradientTape() as tape:
        Px = model(x, training=True)
        if sup:
            loss_value = LLoss(labels,Px)
        else:
            loss_value = aug_loss(Px, x, model, datagen, Lambda=Lambda)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_history.append(loss_value.numpy().mean())


Lbatch = 42
Ubatch = 28
N_batch = 1000
# ,tf.newaxis
def train(model, epochs, Lambda = 1, U = True, L = True):
    for epoch in range(epochs):
        for i in (range(1,N_batch)):
            if(U):
                train_step(Ux[(i-1) * Ubatch:(i) * Ubatch], model,Lambda, sup=False)
            if(L):
                train_step(Lx[(i - 1) * Ubatch:(i) * Ubatch], model, Lambda, sup=True,
                           labels=Ly[(i - 1) * Ubatch:(i) * Ubatch])

        print ('Epoch {} finished'.format(epoch))
        accuracy_history.append(evaluate(Lx[:1000], Ly[:1000], model))

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

train( model = model, epochs = 5, U = False, Lambda = 0.25)
#train( model = model2, epochs = 3, Lambda = 1)
#train( model = model1, epochs = 3, U = False)

#train(U = False)
