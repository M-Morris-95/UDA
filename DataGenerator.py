from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np


def Get_Data(Dataset, validation_split, unlabelled_split):
    if Dataset == 'CIFAR10':
        (x_train, y_train), (x_test, y_test), = tf.keras.datasets.cifar10.load_data()
    elif Dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test), = tf.keras.datasets.mnist.load_data()

    if np.ndim(x_train) == 3:
        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]

    yl_train = np.squeeze(y_train)
    xl_train = x_train / np.max(x_train)

    xu_train = xl_train[unlabelled_split:]
    yl_train = yl_train[:unlabelled_split]
    xl_train = xl_train[:unlabelled_split]


    y_test = np.squeeze(y_test)
    x_test = x_test / np.max(x_test)

    validation_split = int(np.size(y_test)*validation_split)
    x_val = x_test[:validation_split]
    y_val = y_test[:validation_split]

    x_test = x_test[validation_split:]
    y_test = y_test[validation_split:]

    return (xl_train, yl_train, xu_train), (x_test, y_test), (x_val, y_val)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.2,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.2,
    shear_range=0.3,  # set range for random shear
    zoom_range=0.2,  # set range for random zoom
    channel_shift_range=0,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
