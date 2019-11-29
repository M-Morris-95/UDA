import tensorflow as tf
import numpy as np
import pandas as pd


class Semi_Supervised_Trainer:
    def __init__(self, model, args, datagen=[], optimizer=tf.keras.optimizers.Adam()):
        self.model = model
        self.Epochs = args.Epochs
        self.Loss = args.Loss
        self.usup_weight = args.Lambda
        self.n_batch = args.N_Batch
        self.u_n_batch = args.U_Batch
        self.TSA_type = args.TSA
        self.datagen = datagen
        self.optimizer = optimizer
        self.Mode = args.Mode

        self.history_created = False
        self.val_accuracy = np.nan
        self.iter = 0
        self.w_train_acc = 0

    def shuffle(self, x, y=[0]):
        if len(x) == len(y):
            p = np.random.permutation(len(x))
            return x[p], y[p]
        else:
            p = np.random.permutation(len(x))
            return x[p]

    def make_batches(self, xl, yl, xu, shuffle=True):
        if shuffle:
            xl, yl = self.shuffle(xl,yl)
            xu = self.shuffle(xu)

        nl_batches = int(np.ceil(np.shape(xl)[0] / self.n_batch))
        nu_batches = int(np.ceil(np.shape(xu)[0] / self.u_n_batch))

        yl = tf.one_hot(yl, self.num_labels).numpy()
        xl_batch = np.array_split(xl, nl_batches)
        yl_batch = np.array_split(yl, nl_batches)
        xu_batch = np.array_split(xu, nu_batches)

        self.n_batches = min(nl_batches, nu_batches)
        self.total_steps = self.Epochs * self.n_batches

        xl_batch = xl_batch[:self.n_batches]
        yl_batch = yl_batch[:self.n_batches]
        xu_batch = xu_batch[:self.n_batches]


        return xl_batch, yl_batch, xu_batch

    def train(self, model, xl, yl, xu, x_val, y_val):
        self.num_labels = np.size(np.unique(yl))

        for self.epoch in range(self.Epochs):
            xl_batch, yl_batch, xu_batch = self.make_batches(xl, yl, xu, shuffle=True)

            for self.batch in range(self.n_batches):
                self.uda_step(xl_batch[self.batch], yl_batch[self.batch], xu_batch[self.batch])
                self.train_accuracy = self.evaluate(xl_batch[self.batch], yl_batch[self.batch])
                self.create_history(eol=False)

            self.val_accuracy = self.evaluate(x_val, tf.one_hot(y_val, depth=self.num_labels))
            self.create_history(eol=True)

    def _kl_divergence_with_logits(self, p_logits, q_logits):
        p = tf.nn.softmax(p_logits)
        log_p = tf.nn.log_softmax(p_logits)
        log_q = tf.nn.log_softmax(q_logits)

        kl = tf.reduce_sum(p * (log_p - log_q), -1)
        return kl

    def apply_tsa(self, logits, y):
        t_T =  self.iter / self.total_steps

        if self.TSA_type == 'linear':
            at = t_T
        elif self.TSA_type == 'log':
            at = 1 - np.exp(- t_T * 5)
        elif self.TSA_type == 'exponential':
            at = np.exp((t_T - 1) *5)
        else:
            at = 1
        self.TSA_lim = at*(1 - 1 / self.num_labels) + 1 / self.num_labels

        predictions = tf.one_hot(tf.argmax(logits, axis=1), depth=self.num_labels)
        TSA_removals = tf.reduce_max(y * predictions * tf.nn.softmax(logits), axis=1) < self.TSA_lim
        self.loss_s = tf.boolean_mask(self.loss_s, TSA_removals)

    def aug(self, x):
        x_aug = []
        batches = 0
        for batch in self.datagen.flow(x, batch_size=32, shuffle=False):
            x_aug.append(batch)
            batches += 1
            if batches >= len(x) / 32:
                break
        x_aug = np.concatenate(x_aug)
        return(x_aug)

    def uda_step(self, xl, yl, xu):
        xu_aug = self.aug(xu)
        xl_aug = self.aug(xl)

        with tf.GradientTape() as tape:
            logits_xl = self.model(xl_aug, training=True)
            logits_xu_aug = self.model(xu_aug, training=True)
            logits_xu = self.model(xu, training=True)


            self.loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=yl, logits=logits_xl)
            self.apply_tsa(logits_xl, yl)

            if self.Loss == 'KL_D':
                self.loss_u = self._kl_divergence_with_logits(logits_xu, logits_xu_aug)
            else:
                self.loss_u = tf.square(tf.nn.softmax(logits_xu) - tf.nn.softmax(logits_xu_aug))

            self.loss_s = tf.reduce_mean(self.loss_s)
            self.loss_u = tf.reduce_mean(self.loss_u)

            loss = self.loss_s + self.usup_weight * self.loss_u

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        self.optimizer.apply_gradients(zip(grads, var_list))
        return

    def evaluate(self, x_val, y_val, batch_size=32):
        y_pred = self.predict(x_val, batch_size=batch_size)

        accuracy = (tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_val, axis = 1)), tf.float32)) * 100).numpy()
        return accuracy

    def predict(self, x, batch_size=32):
        n_batches = int(np.ceil(np.shape(x)[0] / batch_size))
        x_batches = np.array_split(x, n_batches)
        predictions = []
        for batch in x_batches:
            predictions.append(tf.nn.softmax(self.model(batch)))
        predictions = tf.concat(predictions, axis = 0)
        return predictions



    def create_history(self, eol):
        if not self.history_created:
            self.history_created = True
            self.history = pd.DataFrame(index=range(0, self.total_steps),
                                        columns=['Epoch', 'Batch', 'Iteration', 'Training_Accuracy',
                                                 'Validation_Accuracy',
                                                 'Weighted_Training_Accuracy',
                                                 'Unsupervised_Loss', 'Supervised_Loss'], dtype=float)

        self.w_train_acc = (self.w_train_acc * self.batch + self.train_accuracy) / (self.batch + 1)

        self.history.Epoch[self.iter] = self.epoch + 1
        self.history.Batch[self.iter] = self.batch + 1
        self.history.Iteration[self.iter] = self.iter
        self.history.Training_Accuracy[self.iter] = self.train_accuracy
        self.history.Validation_Accuracy[self.iter] = self.val_accuracy
        self.history.Unsupervised_Loss[self.iter] = self.loss_u
        self.history.Supervised_Loss[self.iter] = self.loss_s
        self.history.Weighted_Training_Accuracy[self.iter] = self.w_train_acc
        if not eol:
            print(
                'Epoch {epc} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, L-divergence:{divL:1.4f}, L-cross entropy:{supL:1.4f}'.format(
                    epc=(self.epoch + 1),
                    batch=(self.batch + 1),
                    nbatch=self.n_batches,
                    acc=self.w_train_acc,
                    divL=np.mean(
                        self.history.Unsupervised_Loss[
                        self.iter - self.batch: self.iter - 1]),
                    supL=np.mean(
                        self.history.Supervised_Loss[
                        self.iter - self.batch: self.iter - 1])
                    ),end='\r')
        else:
            print('Epoch {epc} {batch}/{nbatch}, train accuracy:{acc:1.2f}%, validation accuracy:{valacc:1.2f}%,L-divergence:{divL:1.4f}, L-cross entropy:{supL:1.4f}'.format(epc=(self.epoch + 1),
                                                                                 batch=(self.batch + 1),
                                                                                 nbatch=self.n_batches,
                                                                                 acc=self.w_train_acc,
                                                                                 valacc=self.val_accuracy,
                                                                                 divL=np.mean(self.history.Unsupervised_Loss[self.iter - self.batch: self.iter - 1]),
                                                                                 supL=np.mean(self.history.Supervised_Loss[self.iter - self.batch: self.iter - 1])
                                                                                 ))

        self.iter += 1