import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime


class Semi_Supervised_Trainer:
    def __init__(self, model, args, datagen=[], optimizer=tf.keras.optimizers.Adam()):
        self.model = model
        self.Epochs = args.Epochs
        self.Loss = args.Loss
        self.usup_weight = args.Lambda
        self.n_batch = args.N_Batch
        self.u_n_batch = args.U_Batch
        self.TSA_type = args.TSA
        self.uTSA_type = args.UTSA
        self.datagen = datagen
        self.optimizer = optimizer
        self.Mode = args.Mode

        self.loss_s = 0
        self.loss_u = 0
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
            xl, yl = self.shuffle(xl, yl)
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
            self.val_accuracy = np.nan
            xl_batch, yl_batch, xu_batch = self.make_batches(xl, yl, xu, shuffle=True)
            self.create_history(epc=True)
            self.epoch_start = time.time()
            for self.batch in range(self.n_batches):
                self.batch_start = time.time()
                if self.Mode == 'UDA':
                    self.uda_step(xl_batch[self.batch], yl_batch[self.batch], xu_batch[self.batch])
                if self.Mode == 'Supervised':
                    self.sup_step(xl_batch[self.batch], yl_batch[self.batch])
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
        t_T = self.iter / self.total_steps

        if self.TSA_type == 'linear':
            at = t_T
        elif self.TSA_type == 'log':
            at = 1 - np.exp(- t_T * 5)
        elif self.TSA_type == 'exponential':
            at = np.exp((t_T - 1) * 5)
        else:
            at = 1
        self.TSA_lim = at * (1 - 1 / self.num_labels) + 1 / self.num_labels

        predictions = tf.one_hot(tf.argmax(logits, axis=1), depth=self.num_labels)
        TSA_removals = tf.reduce_max(y * predictions * tf.nn.softmax(logits), axis=1) < self.TSA_lim
        self.loss_s = tf.boolean_mask(self.loss_s, TSA_removals)

    def apply_utsa(self, logits_x):
        t_T = self.iter / self.total_steps

        if self.uTSA_type == 'linear':
            at = t_T
        elif self.uTSA_type == 'log':
            at = 1 - np.exp(- t_T * 5)
        elif self.uTSA_type == 'exponential':
            at = np.exp((t_T - 1) * 5)
        else:
            at = 0
        self.uTSA_lim = at * (1 - 1 / self.num_labels) + 1 / self.num_labels

        uTSA_removals = tf.reduce_max(tf.nn.softmax(logits_x), axis=1) > self.uTSA_lim
        self.loss_u = tf.boolean_mask(self.loss_u, uTSA_removals)

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

    def uda_step(self, xl, yl, xu):
        xu_aug = self.aug(xu)
        xl_aug = self.aug(xl)

        with tf.GradientTape() as tape:
            logits_xl = self.model(xl_aug, training=True)
            logits_xu_aug = self.model(xu_aug, training=True)
            logits_xu = self.model(xu, training=True)

            self.loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=yl, logits=logits_xl)

            if self.Loss == 'KL_D':
                self.loss_u = self._kl_divergence_with_logits(logits_xu, logits_xu_aug)
            else:
                self.loss_u = tf.square(tf.nn.softmax(logits_xu) - tf.nn.softmax(logits_xu_aug))

            self.apply_tsa(logits_xl, yl)
            self.apply_utsa(logits_xu)

            self.loss_s = tf.reduce_mean(self.loss_s)
            self.loss_u = tf.reduce_mean(self.loss_u)

            loss = self.loss_s + self.usup_weight * self.loss_u

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        self.optimizer.apply_gradients(zip(grads, var_list))
        return

    def sup_step(self, xl, yl):
        xl_aug = self.aug(xl)

        with tf.GradientTape() as tape:
            logits_xl = self.model(xl_aug, training=True)

            self.loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=yl, logits=logits_xl)
            self.loss_s = tf.reduce_mean(self.loss_s)
            loss = self.loss_s

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        self.optimizer.apply_gradients(zip(grads, var_list))
        return

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

    def create_history(self, eol=True, epc=False):
        if epc:
            print('Epoch {epc}/{epc_end}'.format(epc=(self.epoch + 1), epc_end=(self.Epochs)))
            return
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

        eq_n = int(30 * (self.batch + 1) / self.n_batches)
        done = str('=') * eq_n + str('>') + str('.') * (29 - eq_n)
        done = done[:30]

        u_loss = np.mean(self.history.Unsupervised_Loss[self.iter - self.batch: self.iter - 1])
        s_loss = np.mean(self.history.Supervised_Loss[self.iter - self.batch: self.iter - 1])
        if u_loss != 0:
            u_loss = str(' - u_loss: {u_loss:1.4f}'.format(u_loss=u_loss))
            s_loss = str(' - s_loss: {s_loss:1.4f}'.format(s_loss=s_loss))
        else:
            u_loss = ''
            s_loss = str(' - loss: {s_loss:1.4f}'.format(s_loss=s_loss))

        acc = str(' - accuracy: {acc:1.2f}%'.format(acc=self.w_train_acc))

        if eol:
            val_acc = str(' - val accuracy: {valacc:1.2f}%'.format(valacc=self.val_accuracy))
            end = str('\n')

            total_t = time.time() - self.epoch_start
            step_t = int(1000 * total_t / self.n_batches)
            time_summary = str(' - {total_t:1.1f}s {step_t}ms/step'.format(total_t=total_t,
                                                                           step_t=step_t))

        else:
            val_acc = str('')
            end = str('\r')

            ETA = (self.n_batches - 1 - self.batch) * (time.time() - self.batch_start)
            ETA = str(datetime.timedelta(seconds=int(ETA)))
            while ETA[0] == ':' or ETA[0] == '0':
                ETA = ETA[1:]
                if ETA == '':
                    break
            time_summary = str('- ETA: {ETA}'.format(ETA=ETA))

        print('{batch}/{nbatch} [{done}]{time_summary}{acc}{val_acc}{u_loss}{s_loss}'.format(
            batch=(self.batch + 1),
            nbatch=self.n_batches,
            done=done,
            time_summary=time_summary,
            acc=acc,
            val_acc=val_acc,
            u_loss=u_loss,
            s_loss=s_loss
        ), end=end)

        self.iter += 1

    def gpu_uda_step(self, xl, yl, xu, xu_aug):

        with tf.GradientTape() as tape:
            logits_xl = self.model(xl, training=True)
            logits_xu_aug = self.model(xu_aug, training=True)
            logits_xu = self.model(xu, training=True)

            self.loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=yl, logits=logits_xl)

            if self.Loss == 'KL_D':
                self.loss_u = self._kl_divergence_with_logits(logits_xu, logits_xu_aug)
            else:
                self.loss_u = tf.square(tf.nn.softmax(logits_xu) - tf.nn.softmax(logits_xu_aug))

            self.apply_tsa(logits_xl, yl)
            self.apply_utsa(logits_xu)

            self.loss_s = tf.reduce_mean(self.loss_s)
            self.loss_u = tf.reduce_mean(self.loss_u)

            loss = self.loss_s + self.usup_weight * self.loss_u

        self.var_list = self.model.trainable_variables
        grads = tape.gradient(loss, self.var_list)
        self.optimizer.apply_gradients(zip(grads, self.var_list))
        return

    def gpu_make_batches(self, xl, yl, xu, xu_aug, shuffle=True):
        if shuffle:
            xl, yl = self.shuffle(xl, yl)
            xu, xu_aug = self.shuffle(xu, xu_aug)

        nl_batches = int(np.ceil(np.shape(xl)[0] / self.n_batch))
        nu_batches = int(np.ceil(np.shape(xu)[0] / self.u_n_batch))


        self.n_batches = min(nl_batches, nu_batches)
        self.total_steps = self.Epochs * self.n_batches

        yl = tf.one_hot(yl, self.num_labels).numpy()
        xl_batch = np.array_split(xl, nl_batches)
        yl_batch = np.array_split(yl, nl_batches)
        xu_batch = np.array_split(xu, nu_batches)
        xu_aug_batch = np.array_split(xu_aug, nu_batches)


        xl_batch = xl_batch[:self.n_batches]
        yl_batch = yl_batch[:self.n_batches]
        xu_batch = xu_batch[:self.n_batches]
        xu_aug_batch = xu_aug_batch[:self.n_batches]

        return xl_batch, yl_batch, xu_batch, xu_aug_batch

    def gpu_train(self, model, xl, yl, xu, x_val, y_val):
        self.num_labels = np.size(np.unique(yl))

        for self.epoch in range(self.Epochs):
            self.val_accuracy = np.nan


            xl_batch, yl_batch, xu_batch = self.make_batches(xl, yl, xu, shuffle=True)

            xl_aug_batch = []
            xu_aug_batch = []
            for i in range(self.n_batches):
                xl_aug_batch.append(self.aug(xl_batch[i]))
                xu_aug_batch.append(self.aug(xu_batch[i]))

            self.create_history(epc=True)
            self.epoch_start = time.time()

            for self.batch in range(self.n_batches):
                self.batch_start = time.time()
                self.gpu_uda_step(xl_aug_batch[self.batch], yl_batch[self.batch], xu_batch[self.batch], xu_aug_batch[self.batch])
                self.train_accuracy = self.evaluate(xl_batch[self.batch], yl_batch[self.batch])
                self.create_history(eol=False)

            self.val_accuracy = self.evaluate(x_val, tf.one_hot(y_val, depth=self.num_labels))
            self.create_history(eol=True)