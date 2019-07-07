# Siamese Neural Network for Keras

This project provides a lightweight, easy to use and flexible siamese neural network module for use with the Keras 
framework. 

Siamese neural networks are used to generate embeddings that describe inter and extra class relationships. 
This makes Siamese Networks like many other similarity learning algorithms suitable as a pre-training step for many 
classification problems.

An example of the siamese network module being used to produce a noteworthy 99.85% validation performance on the MNIST 
dataset with no data augmentation and minimal modification from the Keras example is provided.

## Installation

Create and activate a virtual environment for the project.
```sh
$ virtualenv env
$ source env/bin/activate
```

To install the module directly from GitHub:
```
$ pip install git+https://github.com/aspamers/siamese
```

The module will install keras and numpy but no back-end (like tensorflow). This is deliberate since it leaves the module 
decoupled from any back-end and gives you a chance to install whatever backend you prefer. 

To install tensorflow:
```
$ pip install tensorflow
```

To install tensorflow with gpu support:
```
$ pip install tensorflow-gpu
```

## To run examples

With the activated virtual environment with the installed python package run the following commands.

To run the mnist baseline example:
```
$ python mnist_example.py
```

To run the mnist siamese pretrained example:
```
$ python mnist_siamese_example.py
```

## Usage
For detailed usage examples please refer to the examples and unit test modules. If the instructions are not sufficient 
feel free to make a request for improvements.

- Import the module
```python
from siamese import SiameseNetwork
```

- Load or generate some data.
```python
x_train = np.random.rand(100, 3)
y_train = np.random.randint(num_classes, size=100)

x_test = np.random.rand(30, 3)
y_test = np.random.randint(num_classes, size=30)
```

- Design a base model
```python
def create_base_model(input_shape):
    model_input = Input(shape=input_shape)
    embedding = Flatten()(model_input)
    embedding = Dense(128)(embedding)
    return Model(model_input, embedding)
```

- Design a head model
```python
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
```
- Create an instance of the SiameseNetwork class
```python
base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)
siamese_network = SiameseNetwork(base_model, head_model)
```

- Compile the model
```python
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam())
```

- Train the model
```python
siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=64,
                    epochs=epochs)
```

## Development Environment
Create and activate a test virtual environment for the project.
```sh
$ virtualenv env
$ source env/bin/activate
```

Install requirements
```sh
$ pip install -r requirements.txt
```

Install the backend of your choice.
```
$ pip install tensorflow
```

Run tests
```sh
$ pytest tests/test_siamese.py
```

## Development container
To set up the vscode development container follow the instructions at the link provided:
https://github.com/aspamers/vscode-devcontainer

You will also need to install the nvidia docker gpu passthrough layer:
https://github.com/NVIDIA/nvidia-docker
