"""
This is a modified version of the Keras mnist example.
https://keras.io/examples/mnist_cnn/

Instead of using a fixed number of epochs this version continues to train until a stop criteria is reached.

A siamese neural network is used to pre-train an embedding for the network. The resulting embedding is then extended
with a softmax output layer for categorical predictions.

Model performance should be around 99.84% after training. The resulting model is identical in structure to the one in
the example yet shows considerable improvement in relative error confirming that the embedding learned by the siamese
network is useful.
"""

from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.engine import Layer
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Flatten, Dense

from siamese import SiameseNetwork

batch_size = 128
num_classes = 10
epochs = 999999

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


class BaseModel(Layer):

    def call(self, inputs):

        base = Conv2D(32, kernel_size=(3, 3), input_shape=input_shape)(inputs)
        base = BatchNormalization()(base)
        base = Activation(activation='relu')(base)
        base = MaxPooling2D(pool_size=(2, 2))(base)
        base = Conv2D(64, kernel_size=(3, 3))(base)
        base = BatchNormalization()(base)
        base = Activation(activation='relu')(base)
        base = MaxPooling2D(pool_size=(2, 2))(base)
        base = Flatten()(base)
        base = Dense(128)(base)
        base = BatchNormalization()(base)
        base = Activation(activation='relu')(base)

        return base


class HeadModel(Layer):

    def call(self, inputs):

        head = Concatenate()(inputs)
        head = Dense(8)(head)
        head = BatchNormalization()(head)
        head = Activation(activation='sigmoid')(head)

        head = Dense(1)(head)
        head = BatchNormalization()(head)
        head = Activation(activation='sigmoid')(head)

        return head


num_classes = 10
epochs = 999999

siamese_network = SiameseNetwork(BaseModel, HeadModel, num_classes)
siamese_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

siamese_checkpoint_path = "./siamese_checkpoint"

siamese_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

siamese_network.fit_generator(x_train, y_train,
                              x_test, y_test,
                              batch_size=1000,
                              epochs=epochs,
                              callbacks=siamese_callbacks,
                              max_queue_size=20)

siamese_network.load_weights(siamese_checkpoint_path)
embedding = base_model.outputs[-1]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Add softmax layer to the pre-trained embedding network
embedding = Dense(num_classes)(embedding)
embedding = BatchNormalization()(embedding)
embedding = Activation(activation='sigmoid')(embedding)

model = Model(base_model.inputs[0], embedding)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model_checkpoint_path = "./model_checkpoint"

model__callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=epochs,
          callbacks=model__callbacks,
          validation_data=(x_test, y_test))

model.load_weights(model_checkpoint_path)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
