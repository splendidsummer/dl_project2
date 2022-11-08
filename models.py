import tensorflow as tf
from tensorflow import keras
import configs
from keras import Sequential, Model

from keras.layers import Dense, Activation, Add, Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, UpSampling2D, GlobalAveragePooling2D, Concatenate


data_augmentation = Sequential(
  [
    layers.Resizing(configs.augment_config['img_resize'], configs.augment_config['img_resize']),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal_and_vertical"),

    layers.RandomRotation(configs.augment_config['random_ratio']),
    layers.CenterCrop(configs.augment_config['img_crop'], configs.augment_config['img_crop']),
    layers.RandomZoom(configs.augment_config['random_ratio']),
    # layers.RandomBrightness(configs.img_factor),
    layers.RandomContrast(configs.augment_config['img_factor']),
    layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
    RandomGray(configs.augment_config['random_ratio']),
  ]
)

def build_resnet50():
    base_