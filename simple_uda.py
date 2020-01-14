
from DataGenerator import get_datagen, Get_Data
import parser
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
import tensorflow as tf
import time

parser = parser.GetParser()
args = parser.parse_args()

(xl_train, yl_train, xu_train), (x_test, y_test), (x_val, y_val) = Get_Data(args.Dataset, 0.1, labelled_split = 100)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=xl_train.shape[1:]))
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
model.add(Dense(np.size(np.unique(yl_train)), activation = 'softmax'))

class uda_trainer:
    def __init__(self, model, datagen, labelled_batch_size=8, unlabelled_batch_size = 256):
        self.optimizer = tf.keras.optimizers.Adam()
        self.datagen = datagen
        self.model = model
        self.batch_size = labelled_batch_size
        self.labelled_batch_size = labelled_batch_size
        self.unlabelled_batch_size = unlabelled_batch_size

    def shuffle(self, x, y=[0]):
        if len(x) == len(y):
            p = np.random.permutation(len(x))
            return x[p], y[p]
        else:
            p = np.random.permutation(len(x))
            return x[p]

    def evaluate(self, x_val, y_val, batch_size=32):
        y_pred = self.predict(x_val, batch_size=batch_size)

        accuracy = (tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_val, axis=1)), tf.float32)) * 100).numpy()
        return accuracy

    def predict(self, x, batch_size=32):
        n_batches = int(np.ceil(np.shape(x)[0] / batch_size))
        x_batches = np.array_split(x, n_batches)
        predictions = []
        for batch in x_batches:
            predictions.append(tf.nn.softmax(self.model(batch)))
        predictions = tf.concat(predictions, axis=0)
        return predictions

    def make_batches(self, xl, yl, xu, do_shuffle=True):
        self.num_labels = np.size(np.unique(yl))
        if do_shuffle:
            xl, yl = self.shuffle(xl, yl)
            xu = self.shuffle(xu)

        self.n_batches = int(np.floor(np.min([xl.shape[0]/self.labelled_batch_size, xu.shape[0]/self.unlabelled_batch_size])))

        yl = tf.one_hot(yl, self.num_labels).numpy()

        xl_batch = xl[:self.labelled_batch_size * self.n_batches].reshape(
            (-1, self.labelled_batch_size, xl.shape[1], xl.shape[2], xl.shape[3]))
        yl_batch = yl[:self.labelled_batch_size * self.n_batches].reshape(
            (-1, self.labelled_batch_size, yl.shape[1]))
        xu_batch = xu[:self.unlabelled_batch_size * self.n_batches].reshape(
            (-1, self.unlabelled_batch_size, xu.shape[1], xu.shape[2], xu.shape[3]))


        for i in range(self.n_batches):
            aug = self.aug(xu_batch[i])
            aug = aug[np.newaxis, :, :, :, :]

            xu_batch = np.append(xu_batch, aug, axis=0)
        xu_batch = xu_batch.reshape(self.n_batches, -1, xu.shape[1], xu.shape[2], xu.shape[3])
        x_batch = np.append(xl_batch, xu_batch, axis=1)

        yu_batch = np.zeros((yl_batch.shape[0], xu_batch.shape[1], yl_batch.shape[2]))
        y_batch = np.append(yl_batch, yu_batch, axis=1)

        self.batch_size = y_batch.shape[1]

        y_batch = y_batch.reshape(-1, y_batch.shape[2])
        x_batch = x_batch.reshape(-1, x_batch.shape[2], x_batch.shape[3], x_batch.shape[4])
        return x_batch, y_batch

    def aug(self, x):
        x_aug = []
        batches = 0
        for batch in self.datagen.flow(x, batch_size=32, shuffle=False):
            x_aug.append(batch)
            batches += 1
            if batches >= len(x) / 32:
                break
        x_aug = np.concatenate(x_aug)
        return (x_aug)

    def uda_step(self, xl, yl, xu, xu_aug):

        with tf.GradientTape() as tape:
            logits_xl = self.model(xl, training=True)
            logits_xu_aug = self.model(xu_aug, training=True)
            logits_xu = self.model(xu, training=True)

            self.loss_s = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true=yl, y_pred=tf.nn.softmax(logits_xl)))
            self.loss_u = tf.reduce_mean(tf.losses.KLD(y_true=tf.nn.softmax(logits_xu), y_pred=tf.nn.softmax(logits_xu_aug)))

            loss = self.loss_s + self.loss_u

        self.var_list = self.model.trainable_variables
        grads = tape.gradient(loss, self.var_list)
        self.optimizer.apply_gradients(zip(grads, self.var_list))
        return

    def gpu_train(self, xl, yl, xu, x_val=0, y_val=0):
        self.num_labels = np.size(np.unique(yl))

        for self.epoch in range(3):
            xl_batch, yl_batch, xu_batch = self.make_batches(xl, yl, xu, do_shuffle=True)

            for self.batch in range(self.n_batches):
                # augment
                xu_aug_batch = self.aug(xu_batch[self.batch])

                self.uda_step(xl_batch[self.batch], yl_batch[self.batch], xu_batch[self.batch], xu_aug_batch)

                self.train_accuracy = self.evaluate(xl_batch[self.batch], yl_batch[self.batch])
                self.create_history(eol=False)

            self.val_accuracy = self.evaluate(x_val, tf.one_hot(y_val, depth=self.num_labels))
            self.create_history(eol=True)

uda = uda_trainer(model, get_datagen(args))
X, Y = uda.make_batches(xl_train, yl_train, xu_train)

8+256+256

def loss(y_pred, y_true):
    y_pred_l, y_pred_u, y_pred_au = tf.split(y_pred, [8,256, 256], 0)
    y_true, _= tf.split(y_true, [8,256+256])
    loss_s = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred_l))
    loss_u = tf.reduce_mean(tf.losses.KLD(y_true=y_pred_u, y_pred=y_pred_au))

    return loss_s+loss_u

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=loss)

for epoch in range(10):
    model.fit(X, Y,
              batch_size=520,
              epochs=1,
              shuffle=False)
    print(str(np.mean(np.argmax(model.predict(x_val), axis=1) == y_val)*100)+ '%')

