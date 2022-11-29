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
    preprocess_inputs = keras.applications.resnet.preprocess_input  # [-1, 1]
    # rescale = Rescaling(1/127.5, offset=-1)
    # base_model.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    drop_layer = Dropout(config.drop_rate)
    prediction_layer = Dense(configs.num_classes, activation='softmax')
    inputs = keras.Input(shape=configs.input_shape)
    # out = data_augmentation(inputs)
    out = preprocess_inputs(inputs)
    out = base_model(out, training=False)
    out = global_average_layer(out)
    out = drop_layer(prediction_layer(out))

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
    out = base_model(out, training=False)
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
        weights="imagenet",
        input_shape=configs.input_shape,
    )
    model = build_finetune_dense_model(resnet50, config)
    return model


def build_efficientnetb7():
    efficientnet_b7 = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )
    model = build_finetune_model(efficientnet_b7)
    return model


def build_efficientnetb0(config):
    efficientnet_b0 = keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )
    model = build_finetune_model(efficientnet_b0, config)
    return model


def build_extractor(base_model, target_layer_names, backbone_type='resnet'):  #

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
    else:
        preprocess_inputs = keras.applications.resnet.preprocess_input

    return feature_extractor, preprocess_inputs


def build_extract_finetune(base_model, target_layer_names, backbone_type='resnet'):
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
    else:
        preprocess_inputs = keras.applications.resnet.preprocess_input

    inputs = keras.Input(shape=configs.input_shape)
    # outs = data_augmentation(inputs)
    outs = preprocess_inputs(inputs)
    outs = feature_extractor(outs)
    global_average_layer = GlobalAveragePooling2D()

    outputs = None
    for output in outs:
        output = global_average_layer(output)
        if outputs is not None:
            outputs = Concatenate()([outputs, output])
        else:
            outputs = output

    prediction_layer = Dense(configs.num_classes, activation='softmax')
    out = prediction_layer(outputs)

    model = Model(inputs, out)
    return model


def build_resnet_extractor():
    base_model = keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )

    layer_names = resnet_config.target_layers

    feature_extractor, preprocess_inputs = build_extractor(base_model, layer_names, backbone_type='resnet')
    inputs = keras.Input(shape=configs.input_shape)
    inputs = preprocess_inputs(inputs)
    outs = feature_extractor(inputs)
    feature_extractor = tf.keras.Model(inputs=inputs, outputs=outs)

    return feature_extractor


def build_efficientnetb7_extractor():
    base_model = keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )

    layer_names = efficientnet_config.target_layers

    features_extractor, preprocess_inputs = build_extractor(base_model, layer_names, backbone_type='efficientnet')
    inputs = keras.Input(shape=configs.input_shape)
    inputs = preprocess_inputs(inputs)
    outs = features_extractor(inputs)
    features_extractor = tf.keras.Model(inputs=inputs, outputs=outs)
    return features_extractor


def build_resnet_extract_finetune():
    base_model = keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )

    layer_names = resnet_config.target_layers

    model = build_extract_finetune(base_model, layer_names, backbone_type='resnet')
    return model


def build_efficient_extract_finetune():
    base_model = keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )

    layer_names = efficientnet_config.target_layers
    model = build_extract_finetune(base_model, layer_names, backbone_type='resnet')
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

    pretrain_model = tf.keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )

    print(1111)
    img = tf.random.normal((4, 256, 256, 3))
    # extractor, _ = build_extractor(pretrain_model, resnet_config.target_layers)

    # out = extractor(img)

    resnet_extractor = build_resnet_extractor()
    resnet_outs = resnet_extractor(img)
    efficient_extractor = build_efficientnetb7_extractor()
    eff_outs = efficient_extractor(img)

    model1 = build_resnet_extract_finetune()
    classification1 = model1(img)
    model2 = build_efficient_extract_finetune()
    classification2 = model2(img)

    print(1111)






