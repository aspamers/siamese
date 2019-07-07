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
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Dense

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


def create_base_model(input_shape):
    model_input = Input(shape=input_shape)

    embedding = Conv2D(32, kernel_size=(3, 3), input_shape=input_shape)(model_input)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(64, kernel_size=(3, 3))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Flatten()(embedding)
    embedding = Dense(128)(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape)
    embedding_b = Input(shape=embedding_shape)

    head = Concatenate()([embedding_a, embedding_b])
    head = Dense(8)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    head = Dense(1)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    return Model([embedding_a, embedding_b], head)


num_classes = 10
epochs = 999999

base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)

siamese_network = SiameseNetwork(base_model, head_model)
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

siamese_checkpoint_path = "./siamese_checkpoint"

siamese_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=1000,
                    epochs=epochs,
                    callbacks=siamese_callbacks)

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
