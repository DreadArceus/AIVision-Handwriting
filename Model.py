import tensorflow as tf
from tensorflow import keras
import numpy as np


def train(train_images: np.ndarray, train_labels: np.ndarray) -> None:
    # pixels = (len(train_images[0]), len(train_images[0][0]))
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28), input_shape=(28, 28)),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile()
    model.fit()
