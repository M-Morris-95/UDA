import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale


def LLoss(labels,logits, num_labels = 10):
    OHlabels = tf.one_hot(labels, num_labels)
    sup_loss = OHlabels * -tf.math.log(logits)
    return tf.reduce_mean(sup_loss)

def _kl_divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(p * (log_p - log_q), -1)
    return kl

def aug_loss(Px,x, model, datagen, Lambda = 1):
    x_hat = datagen.flow(x, batch_size=x.shape[0]).next()
    x_hat -= np.min(x_hat)

    Px_hat = model(x_hat)
    KLD = _kl_divergence_with_logits(p_logits=tf.stop_gradient(Px), q_logits=Px_hat)
    KLD = tf.reduce_mean(KLD)
    return KLD

def ULoss(Px,x, model, datagen, Lambda = 0.01):
    # Calculate unsupervised loss

    x_hat = np.squeeze(datagen.flow(x, batch_size=x.shape[0]).next())
    x_hat -= np.min(x_hat)

    Px_hat = model(x_hat)

    A = Px_hat.numpy()
    B = Px.numpy()
    KL_Divergence = tf.transpose(KL_divergence(Px_hat, Px) * tf.ones([10, 1]))

    Exp_Px_hat = tf.transpose(tf.reduce_sum(Px_hat * KL_Divergence, axis=1) * tf.ones([10, 1]))
    Exp_Px = tf.reduce_sum(Px*Exp_Px_hat, axis = 1)

    return Lambda*tf.reduce_mean(Exp_Px)
    #return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(Px - Px_hat),axis = 1)))

def ULoss2(Px,X,model, datagen, Lambda = 0.01):
    Px_hat = model(datagen.flow(X, batch_size=X.shape[0]).next())
    Divergence = (Px - Px_hat)
    Loss = Lambda * tf.reduce_mean(Divergence)
    return(Loss)

def evaluate(x, y_test, model):
    y_pred = model(x)
    y_pred = tf.math.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), tf.float32))*100
    print("Accuracy:{acc:1.2f}%".format(acc=accuracy))
    accuracy = accuracy.numpy()
    return(accuracy)