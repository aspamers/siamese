"""
Siamese neural network module.
"""

import random
import numpy as np

from keras.layers import Input
from keras.models import Model


class SiameseNetwork:
    """
    A simple and lightweight siamese neural network implementation.

    The SiameseNetwork class requires the base and head model to be defined via the constructor. The class exposes
    public methods that allow it to behave similarly to a regular Keras model by passing kwargs through to the
    underlying keras model object where possible. This allows Keras features like callbacks and metrics to be used.
    """
    def __init__(self, base_model, head_model, num_classes):
        """
        Construct the siamese model class with the following structure.

        -------------------------------------------------------------------
        input1 -> base_model |
                             --> embedding --> head_model --> binary output
        input2 -> base_model |
        -------------------------------------------------------------------

        :param base_model: The embedding model.
        * Input shape must be equal to that of data.
        :param head_model: The discriminator model.
        * Input shape must be equal to that of embedding
        * Output shape must be equal to 1..
        :param num_classes: The number of classes in the data
        """
        # Set essential parameters
        self.base_model = base_model
        self.head_model = head_model
        self.num_classes = num_classes

        # Get input shape from base model
        self.input_shape = self.base_model.input_shape[1:]

        # Initialize siamese model
        self.siamese_model = None
        self.__initialize_siamese_model()

    def compile(self, *args, **kwargs):
        """
        Configures the model for training.

        Passes all arguments to the underlying Keras model compile function.
        """
        self.siamese_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the siamese network generator function.

        Redirects arguments to the fit_generator function.
        """
        x_train = args[0]
        y_train = args[1]
        x_test, y_test = kwargs.pop('validation_data')
        batch_size = kwargs.pop('batch_size')

        train_generator = self.__pair_generator(x_train, y_train, batch_size)
        train_steps = max(len(x_train) / batch_size, 1)
        test_generator = self.__pair_generator(x_test, y_test, batch_size)
        test_steps = max(len(x_test) / batch_size, 1)
        self.siamese_model.fit_generator(train_generator,
                                         steps_per_epoch=train_steps,
                                         validation_data=test_generator,
                                         validation_steps=test_steps, **kwargs)

    def fit_generator(self, x_train, y_train, x_test, y_test, batch_size, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the siamese network generator function.

        :param x_train: Training input data.
        :param y_train: Training output data.
        :param x_test: Validation input data.
        :param y_test: Validation output data.
        :param batch_size: Number of pairs to generate per batch.
        """
        train_generator = self.__pair_generator(x_train, y_train, batch_size)
        train_steps = max(len(x_train) / batch_size, 1)
        test_generator = self.__pair_generator(x_test, y_test, batch_size)
        test_steps = max(len(x_test) / batch_size, 1)
        self.siamese_model.fit_generator(train_generator,
                                         steps_per_epoch=train_steps,
                                         validation_data=test_generator,
                                         validation_steps=test_steps,
                                         *args, **kwargs)

    def load_weights(self, checkpoint_path):
        """
        Load siamese model weights. This also affects the reference to the base and head models.

        :param checkpoint_path: Path to the checkpoint file.
        """
        self.siamese_model.load_weights(checkpoint_path)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the siamese network with the same generator that is used to train it. Passes arguments through to the
        underlying Keras function so that callbacks etc can be used.

        Redirects arguments to the evaluate_generator function.

        :return: A tuple of scores
        """
        x = args[0]
        y = args[1]
        batch_size = kwargs.pop('batch_size')

        generator = self.__pair_generator(x, y, batch_size)
        steps = len(x) / batch_size
        return self.siamese_model.evaluate_generator(generator, steps=steps, **kwargs)

    def evaluate_generator(self, x, y, batch_size, *args, **kwargs):
        """
        Evaluate the siamese network with the same generator that is used to train it. Passes arguments through to the
        underlying Keras function so that callbacks etc can be used.

        :param x: Input data
        :param y: Class labels
        :param batch_size: Number of pairs to generate per batch.
        :return: A tuple of scores
        """
        generator = self.__pair_generator(x, y, batch_size=batch_size)
        steps = len(x) / batch_size
        return self.siamese_model.evaluate_generator(generator, steps=steps, *args, **kwargs)

    def __initialize_siamese_model(self):
        """
        Create the siamese model structure using the supplied base and head model.
        """
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        processed_a = self.base_model(input_a)
        processed_b = self.base_model(input_b)

        head = self.head_model([processed_a, processed_b])
        self.siamese_model = Model([input_a, input_b], head)

    def __create_pairs(self, x, class_indices, batch_size):
        """
        Create a numpy array of positive and negative pairs and their associated labels.

        :param x: Input data
        :param class_indices: A python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param batch_size: The number of pair samples to create.
        :return: A tuple of (Numpy array of pairs, Numpy array of labels)
        """
        num_pairs = batch_size / 2
        positive_pairs, positive_labels = self.__create_positive_pairs(x, class_indices, num_pairs)
        negative_pairs, negative_labels = self.__create_negative_pairs(x, class_indices, num_pairs)
        return np.array(positive_pairs + negative_pairs), np.array(positive_labels + negative_labels)

    def __create_positive_pairs(self, x, class_indices, num_positive_pairs):
        """
        Create a list of positive pairs and labels. A positive pair is defined as two input samples of the same class.

        :param x: Input data
        :param class_indices: A python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param num_positive_pairs: The number of positive pair samples to create.
        :return: A tuple of (python list of positive pairs, python list of positive labels)
        """
        positive_pairs = []
        positive_labels = []
        for _ in range(num_positive_pairs):
            class_1 = random.randint(0, self.num_classes - 1)
            num_elements = len(class_indices[class_1])

            index_1, index_2 = self.__randint_unequal(0, num_elements - 1)

            element_index_1, element_index_2 = class_indices[class_1][index_1], class_indices[class_1][index_2]
            positive_pairs.append([x[element_index_1], x[element_index_2]])
            positive_labels.append([1.0])
        return positive_pairs, positive_labels

    def __create_negative_pairs(self, x, class_indices, num_negative_pairs):
        """
        Create a list of negative pairs and labels. A negative pair is defined as two input samples of different class.

        :param x: Input data
        :param class_indices: A python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param num_negative_pairs: The number of negative pair samples to create.
        :return: A tuple of (python list of negative pairs, python list of negative labels)
        """
        negative_pairs = []
        negative_labels = []

        for _ in range(num_negative_pairs):
            cls_1, cls_2 = self.__randint_unequal(0, self.num_classes - 1)

            index_1 = random.randint(0, len(class_indices[cls_1]) - 1)
            index_2 = random.randint(0, len(class_indices[cls_2]) - 1)

            element_index_1, element_index_2 = class_indices[cls_1][index_1], class_indices[cls_2][index_2]
            negative_pairs.append([x[element_index_1], x[element_index_2]])
            negative_labels.append([0.0])
        return negative_pairs, negative_labels

    def __pair_generator(self, x, y, batch_size):
        """
        Creates a python generator that produces pairs from the original input data.
        :param x: Input data
        :param y: Integer class labels
        :param batch_size: The number of pair samples to create per batch.
        :return:
        """
        class_indices = self.__get_class_indices(y)
        while True:
            pairs, labels = self.__create_pairs(x, class_indices, batch_size)

            # The siamese network expects two inputs and one output. Split the pairs into a list of inputs.
            yield [pairs[:, 0], pairs[:, 1]], labels

    def __get_class_indices(self, y):
        """
        Create a python list of lists that contains each of the indices in the input data that belong
        to each class. It is used to find and access elements in the input data that belong to a desired class.
        * Example usage:
        * element_index = class_indices[class][index]
        * element = x[element_index]
        :param y: Integer class labels
        :return: Python list of lists
        """
        return [np.where(y == i)[0] for i in range(self.num_classes)]

    @staticmethod
    def __randint_unequal(lower, upper):
        """
        Get two random integers that are not equal.

        Note: In some cases (such as there being only one sample of a class) there may be an endless loop here. This
        will only happen on fairly exotic datasets though. May have to address in future.
        :param lower: Lower limit inclusive of the random integer.
        :param upper: Upper limit inclusive of the random integer. Need to use -1 for random indices.
        :return: Tuple of (integer, integer)
        """
        int_1 = random.randint(lower, upper)
        int_2 = random.randint(lower, upper)
        while int_1 == int_2:
            int_1 = random.randint(lower, upper)
            int_2 = random.randint(lower, upper)
        return int_1, int_2
