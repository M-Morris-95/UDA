def train_step(self, x, y=np.zeros(3)):
    with tf.GradientTape() as tape:
        predictions = self.model(x, training=True)

        if y.all() == 0:
            loss = self.Lambda * self.divergence_loss(predictions, x)
            self.divergence_loss_history.append(loss)
        else:
            loss = self.categorical_cross_entropy(predictions, y)
            self.supervised_loss_history.append(loss)
            predictions = tf.math.argmax(predictions, axis=1)
            self.batch_accuracy = (tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32)) * 100).numpy()

    var_list = self.model.trainable_variables
    grads = tape.gradient(loss, var_list)
    grads_and_vars = zip(grads, var_list)

    self.optimizer.apply_gradients(grads_and_vars)
    self.loss_history.append(loss.numpy().mean())