from keras import layers
import tensorflow as tf
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


class RandomGray(layers.Layer):
    def __init__(self, p=configs.augment_config['random_ratio'], **kwargs):
        super().__init__(**kwargs)
        self.prob = p

    def call(self, img):
        if tf.random.uniform([]) < self.prob:
            img = tf.image.rgb_to_grayscale(img)
        return img
