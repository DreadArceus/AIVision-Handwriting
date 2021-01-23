import datetime
import tensorflow as tf
from tensorflow import keras
import tensorboard
import numpy as np


def alter_pixels(x, y):
    return x/255.0, y


def process_data(x, y):
    xp = tf.convert_to_tensor(x,dtype=np.float64)
    yp = tf.one_hot(y, depth=10)
    data = tf.data.Dataset.from_tensor_slices((xp, yp)).map(alter_pixels)
    data = data.batch(1000)
    return data


def train_and_test(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray) -> None:
    # pixels = (len(train_images[0]), len(train_images[0][0]))
    train_data = process_data(train_images, train_labels)
    test_data = process_data(test_images, test_labels)
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(optimizer='Adam',
                  loss=keras.losses.MeanSquaredError(), metrics=['accuracy'])
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        train_data.repeat(),
        epochs=30,
        steps_per_epoch=2000,
        validation_data=test_data.repeat(),
        validation_steps=10,
        callbacks=[tb_callback])
    print(history)
