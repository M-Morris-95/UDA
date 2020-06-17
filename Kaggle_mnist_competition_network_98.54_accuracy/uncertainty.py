import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow_probability as tfp

#
# (x_train, y_train), (x_test, y_test), = tf.keras.datasets.mnist.load_data()
# y_test = np.eye(10)[y_train.astype(int)]
# y_test = np.eye(10)[y_train.astype(int)]
# if np.ndim(x_train) == 3:
#     x_train = x_train[:,:,:,np.newaxis]
#     x_test = x_test[:, :, :, np.newaxis]
#
# x_train = x_train/255
# x_test = x_test/255
#
# x_val = x_train[:4200]
# y_val = y_train[:4200]
#
# x_train = x_train[4200:]
# y_train = y_train[4200:]
#
#
#
#
# model=tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(10)
# ])
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               # loss = loss,
#               loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(x_train,y_train, epochs = 1, batch_size = 16, shuffle=True)
#
#
# tfd = tfp.distributions

#
# # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
# def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#     c = np.log(np.expm1(0.1))
#     return tf.keras.Sequential([
#         tfp.layers.VariableLayer(2 * n, dtype=dtype),
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#             tfd.Normal(loc=t[..., :n],
#                        scale=1e-5 + tf.nn.softplus(c+t[..., n:])),
#             reinterpreted_batch_ndims=None))
#     ])
#     '''reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims which will
#     be regarded as event dims. When None all but the first batch axis (batch axis 0) will be
#     transferred to event dimensions (analogous to tf.layers.flatten).'''
#
#
# def prior_trainable(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#
#     return tf.keras.Sequential([
#         tfp.layers.VariableLayer(n, dtype=dtype),
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#             tfd.Normal(loc=t, scale=1),
#             reinterpreted_batch_ndims=None))
#     ])
#
# kl_loss_weight = 1.0 / (x_train.shape[0] / 16)
# y_train = np.eye(2)[y_train.astype(int)]
#
# model=tf.keras.models.Sequential([
#     tfp.layers.DenseVariational(units=32,
#                                 activation=tf.keras.activations.relu,
#                                 make_posterior_fn=posterior_mean_field,
#                                 make_prior_fn=prior_trainable,
#                                 kl_weight=kl_loss_weight),
#     tfp.layers.DenseVariational(units=32,
#                                 activation=tf.keras.activations.relu,
#                                 make_posterior_fn=posterior_mean_field,
#                                 make_prior_fn=prior_trainable,
#                                 kl_weight=kl_loss_weight),
#     tfp.layers.DenseVariational(units=2,
#                                 make_posterior_fn=posterior_mean_field,
#                                 make_prior_fn=prior_trainable,
#                                 kl_weight=kl_loss_weight),
#     tfp.layers.DistributionLambda(
#                     lambda t: tfd.Normal(loc=t[..., :],
#                                          scale=0.01)),
# ])
# loss = lambda y, p_y: -p_y.log_prob(y)
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss = loss,
#               # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(x_train,y_train, epochs = 5000, batch_size = 16, shuffle=True)
#
# pred = np.argmax(model.predict(x_train),1)
#
# plt.scatter(x_train[np.argwhere(pred==True)[:,0],0], x_train[np.argwhere(pred==True)[:,0],1],  color='green', s = 4, alpha = 1, label = 'predicted True')
# plt.scatter(x_train[np.argwhere(pred==False)[:,0],0], x_train[np.argwhere(pred==False)[:,0],1],  color='red', s = 4, alpha = 1, label = 'predicted False')
# plt.plot(np.linspace(0,1,num_points), data(np.linspace(0,1,num_points)),color='black', linewidth=3, label = 'true decision boundary')
# plt.legend()
# plt.show()
#
# print('accuracy = ',np.sum(pred == np.argmax(y_train, 1))/y_train.shape[0]*100,'%')
#
# size= 51
#
# x = np.tile(np.linspace(0,1,size), (size,1)).reshape(-1)
# y = np.repeat(np.linspace(0,1,size), size)
#
# x_test =np.zeros((size*size, 2))
# x_test[:, 0] = x
# x_test[:, 1] = y
# means = []
# for i in range(50):
#     means.append(tf.math.softmax(model(x_test).mean().numpy()).numpy())
#
# pred2 = np.mean(np.asarray(means), 0)
#
# conf = np.max(pred2, 1)
# np.max(conf)
# pred2 = np.argmax(pred2,1)
#
#
# conf = (conf-np.min(conf))/(np.max(conf)-np.min(conf))
# plt.figure(1, figsize = (5.1,5.1))
# for i in range(pred2.shape[0]):
#     if (pred2[i] == 1):
#         plt.scatter(x_test[i,0], x_test[i, 1],
#                     color=[0, 1, 0], marker = 's', edgecolors=None, alpha=conf[i])
#     if (pred2[i] == 0):
#         plt.scatter(x_test[i, 0], x_test[i, 1],
#                     color=[1, 0, 0], marker = 's', edgecolors=None, alpha=conf[i])
# plt.xlim([0,1])
# plt.ylim([0,1])
#
# plt.plot(np.linspace(0, 1, num_points), data(np.linspace(0, 1, num_points)), color='black', linewidth=2,
#              label='true decision boundary')
# plt.show()
#
#
#
#
#
#
#
#
#
# means = []
# for i in range(50):
#     means.append(tf.math.softmax(model(x_test).mean().numpy()).numpy())
#
# conf = np.mean(np.asarray(means), 0)
#
# # conf = (conf-np.min(conf))/(np.max(conf)-np.min(conf))
# plt.figure(1, figsize = (5.1,5.1))
# for i in range(pred2.shape[0]):
#     plt.scatter(x_test[i,0], x_test[i, 1],
#                 color=[conf[i, 0], conf[i, 1], 0], marker = 's', edgecolors=None, alpha=np.max(conf[i, :]))
#
# plt.xlim([0,1])
# plt.ylim([0,1])
#
# plt.plot(np.linspace(0, 1, num_points), data(np.linspace(0, 1, num_points)), color='black', linewidth=2,
#              label='true decision boundary')
# plt.show()