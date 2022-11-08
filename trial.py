import tensorflow as tf
from tensorflow import keras

model = keras.applications.resnet.ResNet50()

print(model.summary())
