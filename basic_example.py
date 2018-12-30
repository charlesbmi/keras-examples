#!/usr/bin/python
"""
Based on "Python for Data Science Cheat Sheet"
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


def main():
    # Generate data
    data = np.random.random((1000,100))
    labels = np.random.randint(2, size=(1000,1))

    # Define neural network model
    model = Sequential()
    model.add(Dense(32,
                    activation='relu',
                    input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png')

    # Create Tensorboard Logs
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()),
                              write_graph=True)

    # Train model
    model.fit(data, labels, epochs=10, batch_size=32, callbacks=[tensorboard])

    # Evaluate model. No separate test set
    predictions = model.predict(data)
    loss, accuracy = model.evaluate(data, labels, batch_size=32)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()
