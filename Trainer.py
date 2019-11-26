import tensorflow as tf
import numpy as np
import progressbar
import pandas as pd

from randaugment import policies as found_policies
from randaugment import augmentation_transforms
aug_policies = found_policies.randaug_policies()
class Network:
    def __init__(self, model, datagen = [], type = 'CIFAR10', optimizer = tf.keras.optimizers.Adam()):
        self.model = model
        self.datagen = datagen
        self.optimizer = optimizer
        self.accuracy = 0
        self.batch_accuracy = 0
        self.type = type
        self.num_categories = 10
        self.console = True

        self.batch = 0
        self.epoch = 0
        self.iteration = 0

    def predict(self, x, batch_size=32):
        n_batches = int(np.ceil(np.shape(x)[0] / batch_size))
        x_batches = np.array_split(x, n_batches)
        predictions = np.array([])
        for batch in x_batches:
            batch_pred = tf.math.argmax(self.model(batch), axis = 1)
            predictions = np.append(predictions, batch_pred.numpy())
        return predictions

    def evaluate(self, x_val, y_val, batch_size = 32):
        y_pred = self.predict(x_val, batch_size = batch_size)
        accuracy = (tf.reduce_mean(tf.cast(tf.equal(y_pred, y_val), tf.float32)) * 100).numpy()
        return accuracy

    def unison_shuffled_copies(self, X, Y):
        assert len(X) == len(Y)
        p = np.random.permutation(len(X))
        return X[p], Y[p]


    def sup_step(self, Lx, Ly):
        with tf.GradientTape() as tape:
            predictions = self.model(Lx, training=True)
            loss = self.categorical_cross_entropy(predictions, Ly, lim=1)
        predictions = tf.math.argmax(predictions, axis=1)

        self.history.Divergence_Loss[self.iteration] = loss.numpy()
        self.history.Training_Accuracy[self.iteration] = (
                    tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()


        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)

        self.optimizer.apply_gradients(zip(grads, var_list))

    def make_batches(self, train_x, train_y, unlabelled_x, labelled_batch_size, unlabelled_batch_size):
        n_batches = int(np.ceil(np.shape(train_x)[0] / labelled_batch_size))
        if not unlabelled_batch_size:
            u_x_batches = np.array_split(unlabelled_x, n_batches)

        else:
            u_n_batches = int(np.ceil(np.shape(unlabelled_x)[0] / unlabelled_batch_size))

            if u_n_batches is n_batches:
                x_batches = np.array_split(train_x, n_batches)
                y_batches = np.array_split(train_y, n_batches)
                u_x_batches = np.array_split(unlabelled_x, n_batches)

            else:
                print(
                    '{} unlabelled training bactches, {} labelled batches. Will reuse samples to match batch numbers'.format(
                        u_n_batches, n_batches))

                if u_n_batches > n_batches:
                    # data_pts_needed = u_n_batches * labelled_batch_size
                    # multiplier = int(np.ceil(data_pts_needed / np.shape(train_x)[0]))
                    # train_y = np.tile(train_y, (multiplier))
                    # train_x = np.tile(train_x, (multiplier, 1, 1, 1))
                    #
                    # train_x, train_y = self.unison_shuffled_copies(train_x, train_y)
                    #
                    # n_batches = int(np.ceil(np.shape(train_x)[0] / labelled_batch_size))
                    # x_batches = np.array_split(train_x, n_batches)[:u_n_batches]
                    # y_batches = np.array_split(train_y, n_batches)[:u_n_batches]
                    # u_x_batches = np.array_split(unlabelled_x, u_n_batches)
                    # n_batches = u_n_batches

                    x_batches = np.array_split(train_x, n_batches)
                    y_batches = np.array_split(train_y, n_batches)
                    u_x_batches = np.array_split(unlabelled_x, u_n_batches)[:n_batches]

                elif u_n_batches < n_batches:
                    data_pts_needed = n_batches * unlabelled_batch_size
                    multiplier = int(np.ceil(data_pts_needed / np.shape(unlabelled_x)[0]))
                    unlabelled_x = np.tile(unlabelled_x, (multiplier, 1, 1, 1))
                    unlabelled_x[np.random.permutation(len(unlabelled_x))]

                    u_n_batches = int(np.ceil(np.shape(unlabelled_x)[0] / unlabelled_batch_size))
                    u_x_batches = np.array_split(unlabelled_x, u_n_batches)[:n_batches]
                    x_batches = np.array_split(train_x, n_batches)
                    y_batches = np.array_split(train_y, n_batches)

        return (x_batches, y_batches, u_x_batches, n_batches)


    def TSA(self, TSA_type):
        t_T =  self.iteration / self.total_steps

        if TSA_type == 'linear':
            at = t_T
        elif TSA_type == 'log':
            at = 1 - np.exp(- t_T * 5)
        elif TSA_type == 'exponential':
            at = np.exp((t_T - 1) *5)
        else:
            return 1

        return at*(1 - 1 / self.num_categories) + 1 / self.num_categories

    def kl_divergence(self, p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        # log_p = tf.nn.log_softmax(p_logits)
        # log_q = tf.nn.log_softmax(q_logits)

        log_p = tf.math.log(p_logits)
        log_q = tf.math.log(q_logits)

        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

    def categorical_cross_entropy(self, predictions, labels, lim = 1, num_labels=10, OneHot = False):

        if not OneHot:
            labels = tf.one_hot(labels, num_labels)

        correct_confidence = tf.reduce_max(labels * predictions, axis=1)
        correct_confidence = tf.squeeze([tf.where(correct_confidence < lim)])

        sup_loss = labels * -tf.math.log(predictions)
        sup_loss = tf.gather(sup_loss, correct_confidence.numpy())

        return tf.reduce_mean(sup_loss)

    def augment(self, x, rangaugment):
        if rangaugment:
            aug_x = np.empty(np.shape(x))
            for i in range(np.shape(x)[0]):
                chosen_policy = aug_policies[np.random.choice(len(aug_policies))]
                aug_image = augmentation_transforms.apply_policy(chosen_policy, x[i])
                if self.type == 'MNIST':
                    temp = aug_image[:, :, 0]
                    aug_x[i] = temp[:, :, np.newaxis]
                elif self.type == 'CIFAR10':
                    aug_x[i] = aug_image
        else:
            aug_x = self.datagen.flow(x, batch_size=32, shuffle=False).next()

        return aug_x


    def divergence_loss(self, predictions, x, randaugment=False):
        aug_x = self.augment(x, randaugment)
        aug_predictions = self.model(aug_x)

        KLD = self.kl_divergence(predictions, aug_predictions)
        return tf.reduce_mean(tf.math.abs(KLD))

    def global_step(self, Ux, Lx, Ly, lim = 1):
        with tf.GradientTape() as tape:
            predictions = self.model(Ux, training=True)
            Uloss = self.Lambda * self.divergence_loss(predictions, Ux)


            predictions = self.model(Lx, training=True)
            Lloss = self.categorical_cross_entropy(predictions, Ly, lim=lim)
            loss = Uloss + Lloss

        predictions = tf.math.argmax(predictions, axis=1)
        self.history.Divergence_Loss[self.iteration] = Uloss.numpy()
        self.history.Supervised_Loss[self.iteration] = Lloss.numpy()
        self.history.Training_Accuracy[self.iteration] = (tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()


        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)

        self.optimizer.apply_gradients(zip(grads, var_list))

    def Sharpen(self, P, T):
        return tf.pow(P, 1 / T) / tf.reshape(tf.reduce_sum(tf.pow(P, 1 / T), axis=1), (-1, 1))

    def MixUp(self, X1, X2, Y1, Y2, Lambda = 0.6):
        Lambda = max([Lambda, 1-Lambda])
        size = min(X1.shape[0], X2.shape[0], Y1.shape[0], Y2.shape[0])
        X = Lambda * X1[:size] + (1 - Lambda) * X2[:size]
        Y = Lambda * Y1[:size] + (1 - Lambda) * Y2[:size]
        return X, Y



    def MixMatch(self, Ux, Lx, Ly, K=3, num_labels = 10):
        Lx = self.datagen.flow(Lx, batch_size=32, shuffle=False).next()
        Ly = tf.one_hot(Ly, num_labels).numpy()

        U = np.zeros([K, np.shape(Ux)[0], np.shape(Ux)[1], np.shape(Ux)[2], np.shape(Ux)[3]])
        Q = []
        for k in range(K):
            U[k] = self.datagen.flow(Ux, batch_size=32, shuffle=False).next()
            Q.append(self.model(U[k], training=True))
        U = U.reshape(-1, U.shape[2], U.shape[3], U.shape[4])

        Q = tf.reduce_mean(tf.stack(Q), axis=0)
        Q = tf.tile(self.Sharpen(Q, T=0.5), [K, 1])

        y = np.row_stack((Ly, Q.numpy()))
        x = np.row_stack((Lx, U))

        x, y = self.unison_shuffled_copies(x, y)

        xl, yl = self.MixUp(Lx, x[:Ly.shape[0]], Ly, y[:Ly.shape[0]], Lambda = 0.6)
        xu, yu = self.MixUp(Ux, x[Ly.shape[0]:], Q, y[Ly.shape[0]:], Lambda = 0.6)
        return(xl, yl, xu, yu)


    def MixMatchStep(self, Ux, Lx, Ly, K=2, num_labels=10):
        Lambda_u = self.Lambda
        xl, yl, xu, yu = self.MixMatch(Ux, Lx, Ly, K=K, num_labels=num_labels)
        with tf.GradientTape() as tape:

            Lpred = self.model(xl, training=True)
            Lloss = self.categorical_cross_entropy(Lpred, yl, lim=1, OneHot = True)

            Upred = self.model(xu, training=True)
            Uloss = tf.reduce_mean(tf.square(yu - Upred))

            loss = Lloss + Lambda_u * Uloss

        predictions = tf.math.argmax(Lpred, axis=1)

        self.history.Divergence_Loss[self.iteration] = Uloss.numpy()
        self.history.Supervised_Loss[self.iteration] = Lloss.numpy()
        self.history.Training_Accuracy[self.iteration] = (tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()


        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)

        self.optimizer.apply_gradients(zip(grads, var_list))


    def train(self, train_x, train_y, unlabelled_x = 0, val_x=[], val_y=[], epochs=10, Lambda=1, labelled_batch_size=32,
              unlabelled_batch_size=[], TSA = False, mode = 'Supervised'):
        self.mode = mode
        self.Lambda = Lambda
        x_batches, y_batches, u_x_batches, n_batches = self.make_batches(train_x, train_y, unlabelled_x,
                                                                         labelled_batch_size, unlabelled_batch_size)

        self.total_steps = n_batches*epochs

        self.history = pd.DataFrame(index=range(0, self.total_steps),
                                    columns=['Epoch', 'Batch', 'Training_Accuracy', 'Validation_Accuracy', 'Weighted_Training_Accuracy',
                                             'Divergence_Loss', 'Supervised_Loss', 'TSA_Limit'], dtype=float)

        for epoch in range(epochs):
            self.accuracy = 0
            for batch in range(n_batches):
                self.history.Epoch[self.iteration] = epoch+1
                self.history.Batch[self.iteration] = batch+1

                nt = self.TSA(TSA)
                self.history.TSA_Limit[self.iteration] = nt

                if self.mode == 'UDA':
                    self.global_step(u_x_batches[batch], x_batches[batch], y_batches[batch], lim = nt)
                elif self.mode == 'MixMatch':
                    self.MixMatchStep(u_x_batches[batch], x_batches[batch], y_batches[batch])
                elif self.mode == 'Supervised':
                    self.sup_step(x_batches[batch], y_batches[batch])
                else:
                    print('undefined training mode, should be UDA, MixMatch, or Supervised')
                    break

                self.accuracy = (self.accuracy * batch + self.history.Training_Accuracy[self.iteration])/(batch+1)
                self.history.Weighted_Training_Accuracy[self.iteration] = self.accuracy

                print('Epoch {epc}/{epc_max} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, '
                      'L-divergence:{divL:1.3f}, L-cross entropy:{supL:1.3f}, TSA limit:{tsa_lim:1.2f}'.format(epc = (epoch + 1),
                                                                                  epc_max = epochs,
                                                                                  batch = (batch + 1),
                                                                                  nbatch = n_batches,
                                                                                  acc = self.accuracy,

                                                                                  divL = self.history.Divergence_Loss[self.iteration],
                                                                                  supL = self.history.Supervised_Loss[self.iteration],
                                                                                  tsa_lim = nt),
                end='\r')

                self.iteration += 1
            if val_x.any():
                accuracy = self.evaluate(val_x, val_y)
                self.history.Validation_Accuracy[self.iteration-1] = accuracy


            print('Epoch {epc} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, validation accuracy:{valacc:1.2f}%, '
                  'L-divergence:{divL:1.4f}, L-cross entropy:{supL:1.4f}, TSA limit:{tsa_lim:1.2f}'.format(epc=(epoch + 1),
                                                                                            batch=(batch + 1),
                                                                                            nbatch=n_batches,
                                                                                            acc=self.accuracy,
                                                                                            valacc=accuracy,
                                                                                            divL=self.history.Divergence_Loss[self.iteration-1],
                                                                                            supL=self.history.Supervised_Loss[self.iteration-1],
                                                                                            tsa_lim=nt))
