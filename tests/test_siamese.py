
"""
Tests for the siamese neural network module
"""

import numpy as np
import keras
from keras import Model, Input
from keras.layers import Concatenate, Dense, BatchNormalization, Activation

from siamese import SiameseNetwork


def test_siamese():
    """
    Test that all components the siamese network work correctly by executing a
    training run against generated data.
    """

    num_classes = 5
    input_shape = (3,)
    epochs = 1000

    # Generate some data
    x_train = np.random.rand(100, 3)
    y_train = np.random.randint(num_classes, size=100)

    x_test = np.random.rand(30, 3)
    y_test = np.random.randint(num_classes, size=30)

    # Define base and head model
    def create_base_model(input_shape):
        model_input = Input(shape=input_shape)

        embedding = Dense(4)(model_input)
        embedding = BatchNormalization()(embedding)
        embedding = Activation(activation='relu')(embedding)

        return Model(model_input, embedding)

    def create_head_model(embedding_shape):
        embedding_a = Input(shape=embedding_shape)
        embedding_b = Input(shape=embedding_shape)

        head = Concatenate()([embedding_a, embedding_b])
        head = Dense(4)(head)
        head = BatchNormalization()(head)
        head = Activation(activation='sigmoid')(head)

        head = Dense(1)(head)
        head = BatchNormalization()(head)
        head = Activation(activation='sigmoid')(head)

        return Model([embedding_a, embedding_b], head)

    # Create siamese neural network
    base_model = create_base_model(input_shape)
    head_model = create_head_model(base_model.output_shape)
    siamese_network = SiameseNetwork(base_model, head_model)

    # Prepare siamese network for training
    siamese_network.compile(loss='binary_crossentropy',
                            optimizer=keras.optimizers.adam())

    # Evaluate network before training to establish a baseline
    score_before = siamese_network.evaluate_generator(
        x_train, y_train, batch_size=64
    )

    # Train network
    siamese_network.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=64,
                        epochs=epochs)

    # Evaluate network
    score_after = siamese_network.evaluate(x_train, y_train, batch_size=64)

    # Ensure that the training loss score improved as a result of the training
    assert(score_before > score_after)
