"""
Here we are going to build models for:
1. Feature Extration and
2. Pretrain & Finetune

And we are going to utilize at least the following backbone networks with s
pre-trained from "imagenet" and "Places"(A 10 million Image Database for Scene Recognition):
1. Resnet50
2. EfficientNe
3. Vision Transformer on Imagenet-21k (need to be more specified)
"""

import tensorflow as tf
from tensorflow import keras
import configs
from tensorflow import keras
from keras import Sequential, Model
from keras import layers
from customized import *

from keras.layers import Dense, Activation, Add, Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, UpSampling2D, GlobalAveragePooling2D, Concatenate, Rescaling


data_augmentation = Sequential([
    layers.Resizing(configs.augment_config['img_resize'], configs.augment_config['img_resize']),
    layers.RandomCrop(configs.augment_config['img_crop_size'], configs.augment_config['img_crop_size']),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(configs.augment_config['random_ratio']),
    layers.CenterCrop(configs.augment_config['img_crop_size'], configs.augment_config['img_crop_size']),
    layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
    layers.RandomZoom(configs.augment_config['random_ratio']),
    # layers.RandomBrightness(configs.img_factor),
    layers.RandomContrast(configs.augment_config['img_factor']),
    layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
    RandomGray(configs.augment_config['random_ratio']),
  ]
)

vgg19 = keras.applications.vgg19.VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=configs.input_shape,
)

resnet50 = keras.applications.resnet.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=configs.input_shape,
)


efficientnet_b7 = tf.keras.applications.efficientnet.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=configs.input_shape,
)

efficientnet_b0 = keras.applications.efficientnet.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=configs.input_shape,
)


def build_finetune_model(base_model):
    preprocess_inputs = keras.applications.resnet.preprocess_input
    # rescale = Rescaling(1/127.5, offset=-1)
    # base_model.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(configs.num_classes, activation='softmax')
    inputs = keras.Input(shape=configs.input_shape)
    out = data_augmentation(inputs)
    out = preprocess_inputs(out)
    out = base_model(out)
    out = global_average_layer(out)
    out = prediction_layer(out)

    model = Model(inputs, out)

    return model


def build_extractor(base_model, target_layer_names, backbone_type='resnet'):

    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[layer.output for layer in base_model.layers if layer.name in target_layer_names],
    )

    if backbone_type == 'resnet':
        preprocess_inputs = keras.applications.resnet.preprocess_input
    elif backbone_type == 'efficientnet':
        preprocess_inputs = keras.applications.efficientnet.preprocess_input
    elif backbone_type == 'visiontransformer':
        preprocess_inputs = None  # Adding later

    inputs = keras.Input(shape=configs.input_shape)
    outs = data_augmentation(inputs)
    outs = preprocess_inputs(outs)
    outs = feature_extractor(outs)
    global_average_layer = GlobalAveragePooling2D()

    outputs = None
    for output in outs:
        output = global_average_layer(output)
        if outputs is not None:
            outputs = Concatenate()([outputs, output], axis=-1)
        else:
            outputs = output

    prediction_layer = Dense(configs.num_classes, activation='softmax')
    out = prediction_layer(outputs)

    model = Model(inputs, out)

    return model






if __name__ == '__main__':
    model = build_finetune_model()
    print(111)
    print(222)

