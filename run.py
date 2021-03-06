from __future__ import absolute_import, division, print_function, unicode_literals

from ssl import Semi_Supervised_Trainer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from DataGenerator import get_datagen, Get_Data
import matplotlib.pyplot as plt
import os
import time
import sys
import parser
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

parser = parser.GetParser()
args = parser.parse_args()

split = [250, 500, 1000,2000,4000]
split = [4000]
t = time.localtime()

owd = os.getcwd()
for s in split:
    (xl_train, yl_train, xu_train), (x_test, y_test), (x_val, y_val) = Get_Data(args.Dataset, 0.1, s)
    datagen = get_datagen(args)
    datagen.fit(xl_train)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=xl_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(np.size(np.unique(yl_train))))
  #  model.load_weights('logging/Dec-02-2019_1313/UDA_Model.hdf5')
    network = Semi_Supervised_Trainer(model, args, datagen=datagen)

    network.gpu_train(model, xl_train, yl_train, xu_train, x_val, y_val)



    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    os.chdir('logging')
    if not os.path.exists(timestamp):
        os.mkdir(timestamp)

    os.chdir(timestamp)

    text_file = open(str(s)+".txt", "w")
    text_file.write(' '.join(sys.argv[1:]))
    text_file.close()

    network.history.to_csv (str(s)+'.csv', index = None, header=True)
    network.model.save('UDA_Model.hdf5')

    network.history.Validation_Accuracy.dropna().plot()
    network.history.Training_Accuracy.dropna().plot()

    plt.savefig(str(s)+'.png')
    plt.clf()
    os.chdir(owd)
