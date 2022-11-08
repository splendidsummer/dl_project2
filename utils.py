import numpy as np
import configs
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os, random
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import shutil, wandb


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def plot_samples(data_df):
    fig = plt.figure(figsize=(20.,20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 5),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.)
                    )

    for ax, medium in zip(grid, configs.data_classes):
        filepath = data_df[data_df.Medium == medium].sample(1).Organized_img_path.values[0]
        img = tf.keras.preprocessing.image.load_img(filepath)
        ax.imshow(img)
        ax.title.label = "hi"
        ax.set_title(medium)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    grid[-1].remove()
    plt.show()


def plot_result(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def set_config(config):
    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.learning_rate
    weight_decay = config.weight_decay
    early_stopping = config.early_stopping
    activation = config.activation
    normalization = config.normalization
    data_augmentation = config.data_augmentation

    return None


if __name__ == '__main__':
    print(11111)