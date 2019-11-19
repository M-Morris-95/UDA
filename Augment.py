from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from progressbar import ProgressBar
pbar = ProgressBar()
from tensorflow.keras.losses import categorical_crossentropy as CCE



import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

mnist = tf.keras.datasets.mnist


(u_x_train, u_y_train), (x_test, y_test) = mnist.load_data()
(u_x_train, x_test) = u_x_train / 255.0, x_test / 255.

num_labelled = 10
classes = np.unique(u_y_train)

l_x_train = np.zeros((num_labelled*len(classes), 28, 28))

l_y_train = np.zeros((num_labelled*len(classes), ))

count = 0

for i in classes:
    for j in range(1000):
        if u_y_train[j] == i:
            l_y_train[count] = i
            l_x_train[count] = u_x_train[j,:,:]

            u_x_train = np.delete(u_x_train, j, axis = 0)
            u_y_train = np.delete(u_y_train, j, axis=0)

            count += 1
            if count%num_labelled == 0:
                break


# reshape to be [samples][width][height][channels]
u_x_train = u_x_train.reshape((u_x_train.shape[0], 28, 28, 1))

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(u_x_train)

# configure batch size and retrieve one batch of images
do = False
if do:
    pbar = ProgressBar()
    ua_x_train = np.zeros((np.shape(u_x_train)))
    batch_size = 10
    i = 0
    for X_batch, y_batch in pbar(datagen.flow(u_x_train, u_y_train, batch_size=batch_size)):
        ua_x_train[i * batch_size:(i + 1) * batch_size, :, :, :] = X_batch
        i += 1
        if i >= np.shape(u_x_train)[0]/batch_size:
            break

    u_y_train.fill(np.nan)




def myloss(labels,logits,image, model, num_labels = 10):
    OHlabels = tf.one_hot(labels, num_labels)
    nplabels = labels.numpy()

    idxs = np.squeeze(np.argwhere(nplabels == 11))
    Nidxs = np.squeeze(np.argwhere(nplabels != 11))

    temp = model(datagen.flow(tf.gather(image, idxs), batch_size=len(idxs)).next())

    sup_loss = CCE(y_true=tf.gather(OHlabels, Nidxs), y_pred=tf.gather(logits, Nidxs))
    usup_loss = CCE(y_true=tf.gather(logits, idxs), y_pred=temp)

    loss = np.zeros(32)
    loss[idxs] = usup_loss
    loss[Nidxs] = sup_loss

    loss = tf.convert_to_tensor(loss)
    loss = tf.reduce_mean(loss)
    loss = tf.dtypes.cast(loss,dtype = tf.float32,name=None)

    return loss




model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])



penalized_loss(noise=output1)], optimizer='rmsprop')

model.compile(optimizer='adam',
              loss = myloss(y_true, y_pred,image, model),
              loss = [lambda y_true,y_pred: myloss(y_true,y_pred,image, model)],
              #loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.compile(loss=[myloss(noise=output2),
              myloss(noise=output1)],
              optimizer='rmsprop')

model.fit(l_x_train, l_y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

