"""
This is a modified version of the Keras mnist example.
https://keras.io/examples/mnist_cnn/

Instead of using a fixed number of epochs this version continues to train
until the stop criteria is reached.

Model performance should be around 99.4% after training.
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Dense

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

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3),
               input_shape=input_shape)(input)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='softmax')(x)
    return Model(input, x)


model = create_base_network(input_shape)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

checkpoint_path = "./checkpoint"

callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(checkpoint_path,
                    monitor='val_acc',
                    save_best_only=True,
                    verbose=0)
]
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

model.load_weights(checkpoint_path)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
