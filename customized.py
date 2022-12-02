from keras import layers
import tensorflow as tf
from tensorflow_addons.optimizers import MultiOptimizer
import configs


# Customized Layer
def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255 - x)
    return x


class RandomInvert(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)


# class RandomGray(layers.Layer):
#     def __init__(self, p=configs.augment_config['random_ratio'], **kwargs):
#         super().__init__(**kwargs)
#         self.prob = p
#
#     def call(self, img):
#         if tf.random.uniform([]) < self.prob:
#             img = tf.image.rgb_to_grayscale(img)
#         return img


class CustomMultiOptimizer(MultiOptimizer):
    def _init_(self, model, layers, optimizers):
        opt_list = []
        for i in range(len(layers)-1):
            c_layers = model.layers[layers[i]:layers[i+1]]
            opt_list.append((optimizers[i], c_layers))
        opt_list.append((optimizers[-1], model.layers[layers[-1]:]))
        print(opt_list)
        super()._init_(opt_list)


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

layer = MyDenseLayer(10)