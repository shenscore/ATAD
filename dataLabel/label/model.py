"""
The file containing implementations to all of the neural network models used in our experiments. These include a LeNet
model for MNIST, a VGG model for CIFAR and a multilayer perceptron model for dicriminative active learning, among others.
"""

import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
import keras

def mirrorTAD(x,y):
    n = x.shape[0]
    H = x[:,25].reshape(n,1)
    V = x[:,24].reshape(n,1)
    vertex_ = x[:,:24]
    vertex = np.flip(vertex_,axis=1)
    rest = x[:,-2:]
    mirror_x = np.hstack((vertex,H,V,rest))
    return(mirror_x,y)


class DiscriminativeEarlyStopping(Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='accuracy', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(e=epoch, a=current))
            self.model.stop_training = True


class DelayedModelCheckpoint(Callback):
    """
    A custom callback for saving the model each time the validation accuracy improves. The custom part is that we save
    the model when the accuracy stays the same as well, and also that we start saving only after a certain amoung of
    iterations to save time.
    """

    def __init__(self, filepath, monitor='val_acc', delay=50, verbose=0, weights=False):

        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == 'val_acc':
            current = logs.get(self.monitor)
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)





def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    """

    # if np.sum(input_shape) < 30:
    if np.sum(input_shape) < 70:
        width = 20
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))
    else:
        width=256
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))

    return model

def get_TAD_model(width=64):
    """
    The MLP model for TAD identification.
    """

    np.random.seed(1)

    # width = 50
    model = Sequential()
    # model.add(keras.Input(shape=(28,)))
    # model.add(keras.Input(shape=(27,)))
    # model.add(keras.Input(shape=(42,)))
    model.add(keras.Input(shape=(28,)))
    model.add(Dense(width, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # model.add(Dense(2, activation='softmax', name='softmax'))
    model.add(Dense(2, activation='sigmoid', name='softmax'))

    np.random.seed(1)

    optimizer = optimizers.Adam(learning_rate=0.0001)
    # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def get_LeNet_model(input_shape, labels=10):
    """
    A LeNet model for MNIST.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='embedding'))
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model



def get_autoencoder_model(input_shape, labels=10):
    """
    An autoencoder for MNIST to be used in the DAL implementation.
    """

    image = Input(shape=input_shape)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = UpSampling2D((2, 2), name='embedding')(encoder)
    decoder = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

    autoencoder = Model(image, decoder)
    return autoencoder


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0],1),dtype='int')
    y_U = np.ones((unlabeled.shape[0],1),dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 20
    optimizer = optimizers.Adam(learning_rate=0.001)
    epochs = 20
    # if np.max(input_shape) == 28:
    #     optimizer = optimizers.Adam(lr=0.0003)
    #     epochs = 20
    # elif np.max(input_shape) == 128:
    #     # optimizer = optimizers.Adam(lr=0.0003)
    #     # epochs = 200
    #     batch_size = 128
    #     optimizer = optimizers.Adam(lr=0.0001)
    #     epochs = 1000 #TODO: was 200
    # elif np.max(input_shape) == 512:
    #     optimizer = optimizers.Adam(lr=0.0002)
    #     # optimizer = optimizers.RMSprop()
    #     epochs = 500
    # elif np.max(input_shape) == 32:
    #     optimizer = optimizers.Adam(lr=0.0003)
    #     epochs = 500
    # else:
    #     optimizer = optimizers.Adam()
    #     # optimizer = optimizers.RMSprop()
    #     epochs = 1000
    #     batch_size = 32

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    # print(Y_train[Y_train==0].shape[0])
    # print(Y_train[Y_train==1].shape[0])
    # print(X_train.shape[0])
    # print(Y_train)
    # exit()
    weight_for_0 = float(X_train.shape[0]) / np.nonzero(Y_train[:,0]==0)[0].shape[0]
    weight_for_1 = float(X_train.shape[0]) / np.nonzero(Y_train[:,0]==1)[0].shape[0]

    model.fit(X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            class_weight={0 : weight_for_0,
                          1 : weight_for_1},
            verbose=2)


    return model

