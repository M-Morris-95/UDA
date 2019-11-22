import tensorflow as tf
import numpy as np
import progressbar


from randaugment import policies as found_policies
from randaugment import augmentation_transforms
aug_policies = found_policies.randaug_policies()
class Network:
    def __init__(self, model, datagen = [], optimizer = tf.keras.optimizers.Adam()):
        self.model = model
        self.datagen = datagen
        self.Lambda = 0
        self.optimizer = optimizer
        self.accuracy = 0
        self.batch_accuracy = 0

        self.accuracy_history = []
        self.divergence_loss_history = [0]
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

    def unison_shuffled_copies(self, X, Y):
        assert len(X) == len(Y)
        p = np.random.permutation(len(X))
        return X[p], Y[p]

    def make_batches(self, train_x, train_y, unlabelled_x, labelled_batch_size, unlabelled_batch_size):
        n_batches = int(np.ceil(np.shape(train_x)[0] / labelled_batch_size))

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

    def kl_divergence(self, p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        # log_p = tf.nn.log_softmax(p_logits)
        # log_q = tf.nn.log_softmax(q_logits)

        log_p = tf.math.log(p_logits)
        log_q = tf.math.log(q_logits)

        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

    def categorical_cross_entropy(self, predictions, labels, lim = 1, num_labels=10):

        OHlabels = tf.one_hot(labels, num_labels)

        correct_confidence = tf.reduce_max(OHlabels * predictions, axis=1)
        correct_confidence = tf.squeeze([tf.where(correct_confidence < lim)])

        sup_loss = OHlabels * -tf.math.log(predictions)
        sup_loss = tf.gather(sup_loss, correct_confidence.numpy())

        return tf.reduce_mean(sup_loss)


    def divergence_loss(self, predictions, x):
        # aug_x = self.datagen.flow(x, batch_size=x.shape[0]).next()
        # aug_x -= np.min(aug_x)
        aug_x = np.empty(np.shape(x))
        for i in range(np.shape(x)[0]):
            chosen_policy = aug_policies[np.random.choice(len(aug_policies))]
            aug_image = augmentation_transforms.apply_policy(chosen_policy, x[i])
            # aug_image /= (np.max(aug_image) - np.min(aug_image))
            # aug_image -= np.min(aug_image)
            aug_x[i] = aug_image


        aug_predictions = self.model(aug_x)
        KLD = self.kl_divergence(predictions, aug_predictions)
        KLD = tf.reduce_mean(KLD)
        return KLD

    def global_step(self, Ux, Lx, Ly, lim = 1):
        with tf.GradientTape() as tape:
            predictions = self.model(Ux, training=True)
            Uloss = self.Lambda * self.divergence_loss(predictions, Ux)
            self.divergence_loss_history.append(Uloss.numpy())

            predictions = self.model(Lx, training=True)
            Lloss = self.categorical_cross_entropy(predictions, Ly, lim=lim)
            self.supervised_loss_history.append(Lloss.numpy())
            predictions = tf.math.argmax(predictions, axis=1)
            self.batch_accuracy = (tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()

            loss = Uloss + Lloss

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)

        self.optimizer.apply_gradients(zip(grads, var_list))

    def sup_step(self, Lx, Ly):
        with tf.GradientTape() as tape:
            predictions = self.model(Lx, training=True)
            loss = self.categorical_cross_entropy(predictions, Ly, lim=1)
            self.supervised_loss_history.append(loss.numpy())
            predictions = tf.math.argmax(predictions, axis=1)
            self.batch_accuracy = (tf.reduce_mean(tf.cast(tf.equal(predictions, Ly), tf.float32)) * 100).numpy()

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)

        self.optimizer.apply_gradients(zip(grads, var_list))

    def TSA(self, steps, TSA_type):
        t_T =  steps / self.total_steps

        if TSA_type == 'linear':
            at = t_T
        elif TSA_type == 'log':
            at = 1 - np.exp(- t_T * 5)
        elif TSA_type == 'exponential':
            at = np.exp((t_T - 1) *5)
        else:
            return 1

        return at*(1 - 1 / self.num_categories) + 1 / self.num_categories

    def train(self, train_x, train_y, unlabelled_x = 0, val_x=[], val_y=[], epochs=10, Lambda=1, labelled_batch_size=32,
              unlabelled_batch_size=[], TSA = False, usup = True):
        self.usup = usup
        self.Lambda = Lambda
        x_batches, y_batches, u_x_batches, n_batches = self.make_batches(train_x, train_y, unlabelled_x,
                                                                         labelled_batch_size, unlabelled_batch_size)
        self.num_categories = 10
        self.total_steps = n_batches*epochs
        for epoch in range(epochs):
            self.accuracy = 0
            steps = epoch*n_batches
            for batch in range(n_batches):
                steps = steps+1

                ## do training sample annealing to reduce overfitting - very hacky
                nt = self.TSA(steps, TSA)
                if self.usup:
                    self.global_step(u_x_batches[batch], x_batches[batch], y_batches[batch], lim = nt)
                else:
                    self.sup_step(x_batches[batch], y_batches[batch])
                self.accuracy = (self.accuracy * batch + self.batch_accuracy)/(batch+1)
                print('Epoch {epc}/{epc_max} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, '
                      'L-divergence:{divL:1.3f}, L-cross entropy:{supL:1.3f}, TSA limit:{tsa_lim:1.2f}'.format(epc = (epoch + 1),
                                                                                  epc_max = epochs,
                                                                                  batch = (batch + 1),
                                                                                  nbatch = n_batches,
                                                                                  acc = self.accuracy,
                                                                                  divL = self.divergence_loss_history[-1],
                                                                                  supL = self.supervised_loss_history[-1],
                                                                                  tsa_lim = nt),
                end='\r')

            if val_x.any():
                accuracy = self.evaluate(val_x, val_y)
                self.accuracy_history.append(accuracy)

            print('Epoch {epc} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, validation accuracy:{valacc:1.2f}%, '
                  'L-divergence:{divL:1.4f}, L-cross entropy:{supL:1.4f}, TSA limit:{tsa_lim:1.2f}'.format(epc=(epoch + 1),
                                                                                 batch=(batch + 1),
                                                                                 nbatch=n_batches,
                                                                                 acc=self.accuracy,
                                                                                 valacc=accuracy,
                                                                                 divL=self.divergence_loss_history[-1],
                                                                                 supL=self.supervised_loss_history[-1],
                                                                                 tsa_lim=nt))