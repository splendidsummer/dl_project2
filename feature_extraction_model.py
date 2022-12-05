import tensorflow as tf
from tensorflow import keras
import configs
from utils import *
import resnet_config
from customized import *
import efficientnet_config
from keras.layers import Dense, Activation, Add, Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, UpSampling2D, GlobalAveragePooling2D, Concatenate, Rescaling

from keras import Sequential, Model
from keras import layers

#
# def build_extractor(base_model, target_layer_names, backbone_type='resnet'):  #
#
#     feature_extractor = tf.keras.Model(
#         inputs=base_model.inputs,
#         outputs=[layer.output for layer in base_model.layers if layer.name in target_layer_names],
#     )
#
#     if backbone_type == 'resnet':
#         preprocess_inputs = keras.applications.resnet.preprocess_input
#     elif backbone_type == 'efficientnet':
#         preprocess_inputs = keras.applications.efficientnet.preprocess_input
#     elif backbone_type == 'visiontransformer':
#         preprocess_inputs = None  # Adding later
#     else:
#         preprocess_inputs = keras.applications.resnet.preprocess_input
#
#     return feature_extractor, preprocess_inputs
#
# def build_resnet_extractor():
#     base_model = keras.applications.resnet.ResNet50(
#         include_top=False,
#         weights="imagenet",
#         input_shape=configs.input_shape,
#     )

#     layer_names = resnet_config.target_layers
#
#     feature_extractor, preprocess_inputs = build_extractor(base_model, layer_names, backbone_type='resnet')
#     inputs = keras.Input(shape=configs.input_shape)
#     inputs = preprocess_inputs(inputs)
#     outs = feature_extractor(inputs)
#     feature_extractor = tf.keras.Model(inputs=inputs, outputs=outs)
#
#     return feature_extractor
#
#
# def build_efficientnetb7_extractor():
#     base_model = keras.applications.efficientnet.EfficientNetB7(
#         include_top=False,
#         weights="imagenet",
#         input_shape=configs.input_shape,
#     )
#
#     layer_names = efficientnet_config.target_layers
#
#     features_extractor, preprocess_inputs = build_extractor(base_model, layer_names, backbone_type='efficientnet')
#     inputs = keras.Input(shape=configs.input_shape)
#     inputs = preprocess_inputs(inputs)
#     outs = features_extractor(inputs)
#     features_extractor = tf.keras.Model(inputs=inputs, outputs=outs)
#     return features_extractor


def build_extract_finetune(base_model, target_layer_names, backbone_type='resnet'):
    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[layer.output for layer in base_model.layers if layer.name in target_layer_names],
    )
    feature_extractor.trainable = False

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


def build_resnet50_extract_finetune():
    base_model = keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=configs.input_shape,
    )
    weight_file = '/root/autodl-tmp/dl_project2/weights/'
    base_model.load_weights(weight_file)

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

    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=configs.input_shape,
    )

    target_layers = efficientnet_config.target_layers
    model = build_extract_finetune(base_model, target_layers)
    model.summary()

    print(111)



