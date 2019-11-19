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

datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

# Fetch and format the mnist data
(x_train, y_train), (x_test, y_test), = tf.keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255
datagen.fit(x_test.reshape((x_test.shape[0], 28, 28, 1)))

split = 0.01
datset_size = 60000
size = np.floor(datset_size*split)
size = int(size)
Lx = x_train[:size]
Ly = y_train[:size]

Ux = x_train[size:]
Uy = y_train[size:]

Ux = Ux[:,:,:,np.newaxis]
Lx = Lx[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]

'''
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
'''
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

def train_step(x, labels, model, Lambda = 1, sup = True, ):
    with tf.GradientTape() as tape:
        Px = model(x, training=True)
        if sup:
            loss_value = LLoss(labels,Px)
        else:
            loss_value = aug_loss(Px, x, model, datagen, Lambda=Lambda)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_history.append(loss_value.numpy().mean())

Lbatch = 32
Ubatch = int(np.floor(32/split - 32))
N_batch = int(np.floor(datset_size / (Ubatch+Lbatch)))
# ,tf.newaxis
def train(model, epochs, Lambda = 1, U = True, L = True):
    for epoch in range(epochs):
        for i in (range(1,N_batch)):
            if(U):
                train_step(Ux[(i-1) * Ubatch:(i) * Ubatch], Uy[(i-1) * Ubatch:(i) * Ubatch], model,Lambda, sup=False)
            if(L):
                train_step(Lx[(i-1) * Lbatch:(i) * Lbatch], Ly[(i-1) * Lbatch:(i) * Lbatch], model,Lambda, sup=True)
        print ('Epoch {} finished'.format(epoch))
        accuracy_history.append(evaluate(x_test[:1000], y_test[:1000], model))


train( model = model, epochs = 10, Lambda = 0.5, U = True)
#train( model = model2, epochs = 3, Lambda = 1)
#train( model = model1, epochs = 3, U = False)

#train(U = False)
