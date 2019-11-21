import tensorflow as tf
import numpy as np
import progressbar

class Network:
    def __init__(self, model, datagen = [], optimizer = tf.keras.optimizers.Adam()):
        self.model = model
        self.accuracy_history = []
        self.loss_history = []
        self.datagen = datagen
        self.Lambda = 0
        self.optimizer = optimizer
        self.accuracy = 0
        self.batch_accuracy = 0
        self.divergence_loss_history = []
        self.supervised_loss_history = []

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

    def kl_divergence(self, p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        log_p = tf.nn.log_softmax(p_logits)
        log_q = tf.nn.log_softmax(q_logits)

        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

    def categorical_cross_entropy(self, predictions, labels, num_labels=10):
        OHlabels = tf.one_hot(labels, num_labels)
        sup_loss = OHlabels * -tf.math.log(predictions)
        return tf.reduce_mean(sup_loss)


    def divergence_loss(self, predictions, x):
        aug_x = self.datagen.flow(x, batch_size=x.shape[0]).next()
        aug_x -= np.min(aug_x)

        aug_predictions = self.model(aug_x)
        KLD = self.kl_divergence(predictions, aug_predictions)
        KLD = tf.reduce_mean(KLD)
        return KLD

    def global_step(self, Ux, Lx, Ly):
        with tf.GradientTape() as tape:
            predictions = self.model(Ux, training=True)
            Uloss = self.Lambda * self.divergence_loss(predictions, Ux)
            self.divergence_loss_history.append(Uloss)

            predictions = self.model(Lx, training=True)
            Lloss = self.categorical_cross_entropy(predictions, Ly)
            self.supervised_loss_history.append(Lloss)
            predictions = tf.math.argmax(predictions, axis=1)
            self.batch_accuracy = (tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()

            loss = Uloss + Lloss

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        grads_and_vars = zip(grads, var_list)

        self.optimizer.apply_gradients(grads_and_vars)
        self.loss_history.append(loss.numpy().mean())



    def unison_shuffled_copies(self, X, Y):
        assert len(X) == len(Y)
        p = np.random.permutation(len(X))
        return X[p], Y[p]

    def make_batches(self, train_x, train_y, unlabelled_x, labelled_batch_size, unlabelled_batch_size):
        n_batches = int(np.ceil(np.shape(train_x)[0] / labelled_batch_size))

        if unlabelled_x.any():
            if not unlabelled_batch_size:
                u_x_batches = np.array_split(unlabelled_x, n_batches)

            else:
                u_n_batches = int(np.ceil(np.shape(unlabelled_x)[0] / unlabelled_batch_size))
                u_x_batches = np.array_split(unlabelled_x, u_n_batches)

                if u_n_batches is n_batches:
                    x_batches = np.array_split(train_x, n_batches)
                    y_batches = np.array_split(train_y, n_batches)

                else:
                    print(
                        '{} unlabelled training bactches, {} labelled batches. Will reuse samples to match batch numbers'.format(
                            u_n_batches, n_batches))

                    if u_n_batches > n_batches:
                        data_pts_needed = u_n_batches * labelled_batch_size
                        multiplier = int(np.ceil(data_pts_needed / np.shape(train_x)[0]))
                        train_y = np.tile(train_y, (multiplier))
                        train_x = np.tile(train_x, (multiplier, 1, 1, 1))

                        train_x, train_y = self.unison_shuffled_copies(train_x, train_y)

                        n_batches = int(np.ceil(np.shape(train_x)[0] / labelled_batch_size))
                        x_batches = np.array_split(train_x, n_batches)[:u_n_batches]
                        y_batches = np.array_split(train_y, n_batches)[:u_n_batches]
                        n_batches = u_n_batches

                    elif u_n_batches < n_batches:
                        data_pts_needed = n_batches * unlabelled_batch_size
                        multiplier = int(np.ceil(data_pts_needed / np.shape(unlabelled_x)[0]))
                        unlabelled_x = np.tile(unlabelled_x, (multiplier, 1, 1, 1))
                        unlabelled_x[np.random.permutation(len(unlabelled_x))]

                        u_n_batches = int(np.ceil(np.shape(unlabelled_x)[0] / unlabelled_batch_size))
                        u_x_batches = np.array_split(unlabelled_x, u_n_batches)[:n_batches]

        return (x_batches, y_batches, u_x_batches, n_batches)

    def train(self, train_x, train_y, unlabelled_x=[], val_x=[], val_y=[], epochs=10, Lambda=1, labelled_batch_size=32,
              unlabelled_batch_size=[]):
        self.Lambda = Lambda
        x_batches, y_batches, u_x_batches, n_batches = self.make_batches(train_x, train_y, unlabelled_x,
                                                                         labelled_batch_size, unlabelled_batch_size)


        for epoch in range(epochs):
            self.accuracy = 0
            for batch in range(n_batches):
                #self.train_step(u_x_batches[batch])
                #self.train_step(x_batches[batch], y_batches[batch])
                self.global_step(u_x_batches[batch], x_batches[batch], y_batches[batch])
                self.accuracy = (self.accuracy * batch + self.batch_accuracy)/(batch+1)
                print('Epoch {}, train accuracy:{acc:1.2f}%, batch: {}/{}'.format(epoch + 1, batch + 1, n_batches, acc = self.accuracy), end='\r')

            if val_x.any():
                accuracy = self.evaluate(val_x, val_y)
                self.accuracy_history.append(accuracy)

            print('Epoch {}, train accuracy:{acc:1.2f}%, validation accuracy:{valacc:1.2f}, batch: {}/{}'.format(
                epoch + 1, batch + 1, n_batches, valacc=accuracy, acc=self.accuracy))