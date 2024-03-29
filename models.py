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
import resnet_config
import efficientnet_config

from keras.layers import Dense, Activation, Add, Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, UpSampling2D, GlobalAveragePooling2D, Concatenate, Rescaling

#
# data_augmentation = Sequential([
#     layers.Resizing(configs.augment_config['img_resize'], configs.augment_config['img_resize']),
#     layers.RandomCrop(configs.augment_config['img_crop_size'], configs.augment_config['img_crop_size']),
#     layers.RandomFlip("horizontal"),
#     layers.RandomFlip("vertical"),
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(configs.augment_config['random_ratio']),
#     layers.CenterCrop(configs.augment_config['img_crop_size'], configs.augment_config['img_crop_size']),
#     layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
#     layers.RandomZoom(configs.augment_config['random_ratio']),
#     # layers.RandomBrightness(configs.img_factor),
#     layers.RandomContrast(configs.augment_config['img_factor']),
#     layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
#     RandomGray(configs.augment_config['random_ratio']),
#   ]
# )


def build_finetune_model(base_model, config):
    base_model.trainable = False
    if configs.wandb_config['architecture'] == 'resnet50':
        preprocess_inputs = keras.applications.resnet.preprocess_input  # [-1, 1]
    elif configs.wandb_config['architecture'] == 'efficientnetb7':
        preprocess_inputs = keras.applications.efficientnet.preprocess_input  # [-1, 1]
    # rescale = Rescaling(1/127.5, offset=-1)
    # base_model.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    drop_layer = Dropout(config.drop_rate)
    prediction_layer = Dense(configs.num_classes, activation='softmax')
    inputs = keras.Input(shape=configs.input_shape)
    # out = data_augmentation(inputs)
    out = preprocess_inputs(inputs)
    # out = base_model(out, training=False)
    out = base_model(out)
    out = drop_layer(global_average_layer(out))
    out = prediction_layer(out)

    model = Model(inputs, out)

    return model


def build_finetune_dense_model(base_model, config):
    base_model.trainable = False
    preprocess_inputs = keras.applications.resnet.preprocess_input  # [-1, 1]
    # rescale = Rescaling(1/127.5, offset=-1)
    # base_model.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    hidden_layer = Dense(config.num_hidden, config.activation)
    drop_layer = Dropout(config.drop_rate)
    prediction_layer = Dense(configs.num_classes, activation='softmax')
    inputs = keras.Input(shape=configs.input_shape)
    # out = data_augmentation(inputs)
    out = preprocess_inputs(inputs)
    out = base_model(out)
    out = global_average_layer(out)
    out = drop_layer(hidden_layer(out))
    out = prediction_layer(out)
    model = Model(inputs, out)

    return model


def build_resnet50(config):
    resnet50 = keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )
    model = build_finetune_model(resnet50, config)
    return model


def build_resnet50_hidden(config):
    resnet50 = keras.applications.resnet.ResNet50(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    resnet50.load_weights(weight_file)
    model = build_finetune_dense_model(resnet50, config)
    return model


def build_efficientnetb7(config):
    efficientnet_b7 = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )

    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb7_notop.h5'
    efficientnet_b7.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b7, config)
    return model


def build_efficientnetb7_hidden(config):
    efficientnet_b7 = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )

    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb7_notop.h5'
    efficientnet_b7.load_weights(weight_file)

    model = build_finetune_dense_model(efficientnet_b7, config)
    return model


def build_efficientnetb6(config):
    efficientnet_b6 = tf.keras.applications.efficientnet.EfficientNetB6(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )

    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb6_notop.h5'
    efficientnet_b6.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b6, config)
    return model


def build_efficientnetb0(config):
    efficientnet_b0 = keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb0_notop.h5'
    efficientnet_b0.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b0, config)
    return model


def build_efficientnetb0_hidden(config):
    efficientnet_b0 = keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb0_notop.h5'
    efficientnet_b0.load_weights(weight_file)

    model = build_finetune_dense_model(efficientnet_b0, config)
    return model


def build_efficientnetb1(config):
    efficientnet_b1 = keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb1_notop.h5'
    efficientnet_b1.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b1, config)
    return model


def build_efficientnetb1_hidden(config):
    efficientnet_b1 = keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb1_notop.h5'
    efficientnet_b1.load_weights(weight_file)

    model = build_finetune_dense_model(efficientnet_b1, config)
    return model


def build_efficientnetb2(config):
    efficientnet_b2 = keras.applications.efficientnet.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb2_notop.h5'
    efficientnet_b2.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b2, config)
    return model


def build_efficientnetb3(config):
    efficientnet_b3 = keras.applications.efficientnet.EfficientNetB3(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb3_notop.h5'
    efficientnet_b3.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b3, config)
    return model


def build_efficientnetb4(config):
    efficientnet_b4 = keras.applications.efficientnet.EfficientNetB4(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb4_notop.h5'
    efficientnet_b4.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b4, config)
    return model


def build_efficientnetb5(config):
    efficientnet_b5 = keras.applications.efficientnet.EfficientNetB5(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb5_notop.h5'
    efficientnet_b5.load_weights(weight_file)

    model = build_finetune_model(efficientnet_b5, config)
    return model


if __name__ == '__main__':
    # model = tf.keras.applications.resnet.ResNet50(include_top=True, weights="imagenet")
    # model.summary()
    # layer_names = [layer.name for layer in model.layers]
    # print(layer_names)
    # model = tf.keras.applications.resnet.ResNet50(include_top=False, weights="imagenet")
    # layer_names = [layer.name for layer in model.layers]
    # print(layer_names.index('conv2_block1_1_conv'))
    # print(layer_names.index('conv5_block3_1_conv'))
    # print(111)

    # build_efficientnetb0()
    # build_efficientnetb7()

    efficientnet_b1 = tf.keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights=None,
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/efficientnetb1_notop.h5'
    efficientnet_b1.load_weights(weight_file)

    layer_names = [layer.name for layer in efficientnet_b1.layers]
    print(layer_names)
    layer_names.index('1111')

    # pretrain_model = tf.keras.applications.resnet.ResNet50(
    #     include_top=True,
    #     weights="imagenet",
    #     input_shape=configs.input_shape,
    # )

    # print(1111)
    # img = tf.random.normal((4, 256, 256, 3))
    # extractor, _ = build_extractor(pretrain_model, resnet_config.target_layers)

    # out = extractor(img)
    #
    # resnet_extractor = build_resnet_extractor()
    # resnet_outs = resnet_extractor(img)
    # efficient_extractor = build_efficientnetb7_extractor()
    # eff_outs = efficient_extractor(img)
    #
    # model1 = build_resnet_extract_finetune()
    # classification1 = model1(img)
    # model2 = build_efficient_extract_finetune()
    # classification2 = model2(img)
    #
    # print(1111)
    # last_index =
    # last2_index =
    # last3_index =
    # last4_index =




