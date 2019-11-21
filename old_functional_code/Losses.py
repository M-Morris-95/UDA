import tensorflow as tf
import numpy as np
from progressbar import ProgressBar
from sklearn.preprocessing import scale


def LLoss(labels,logits, num_labels = 10):
    OHlabels = tf.one_hot(labels, num_labels)
    sup_loss = OHlabels * -tf.math.log(logits)
    return tf.reduce_mean(sup_loss)

def kl_divergence(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(p * (log_p - log_q), -1)
    return kl

def ULoss(Px,x, model, datagen, Lambda = 1):
    x_hat = datagen.flow(x, batch_size=x.shape[0]).next()
    x_hat -= np.min(x_hat)

    Px_hat = model(x_hat)
    KLD = kl_divergence(p_logits=tf.stop_gradient(Px), q_logits=Px_hat)
    KLD = tf.reduce_mean(KLD)
    return KLD

def evaluate(x, y_test, model):
    y_pred = model(x)
    y_pred = tf.math.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), tf.float32))*100

    accuracy = accuracy.numpy()
    return(accuracy)

def train_step(x, labels, model, datagen, loss_history, Lambda = 1, sup = True, optimizer = tf.keras.optimizers.RMSprop()):
    with tf.GradientTape() as tape:
        Px = model(x, training=True)
        if sup:
            loss_value = LLoss(labels,Px)
        else:
            loss_value = ULoss(Px, x, model, datagen, Lambda=Lambda)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_history.append(loss_value.numpy().mean())


def train(model, Ux, Lx, Ly, val_x, val_y, epochs, loss_history, accuracy_history, datagen, Lambda=1, U=True, L=True, Lbatch=32, ):
    N_batch = int(np.floor(np.size(Ly) / Lbatch))
    Ubatch = int(np.floor(np.shape(Ux)[0]/N_batch))

    pbar = ProgressBar()


    for epoch in range(epochs):
        print('Training epoch {}'.format(epoch + 1))
        for i in pbar(range(1, N_batch)):
            if (U):
                train_step(Ux[(i - 1) * Ubatch:(i) * Ubatch], 0, model, datagen,
                           loss_history, Lambda, sup=False)
            if (L):
                train_step(Lx[(i - 1) * Lbatch:(i) * Lbatch], Ly[(i - 1) * Lbatch:(i) * Lbatch], model, datagen,
                           loss_history, Lambda, sup=True)
        accuracy = evaluate(val_x[:1000], val_y[:1000], model)
        accuracy_history.append(accuracy)
        print('Epoch {} finished, Accuracy = {acc:1.2f}%'.format(epoch + 1, acc=accuracy))

