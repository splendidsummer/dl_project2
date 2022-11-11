import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras.layers as layers
import numpy as np
from keras.layers import Conv2D, Concatenate, BatchNormalization, Dense, Activation, Add, MaxPooling2D, Flatten, \
    Dropout, LayerNormalization, UpSampling2D, GlobalAveragePooling2D, Concatenate, Rescaling


class PatchEmbed(keras.layers.Layer):

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)

        assert img_size % patch_size == 0, "img_size mismatches patch_size!!!"

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Acquiring Patch embedding for input image

        self.projection = Conv2D(filters=embed_dim, kernel_size=patch_size,
                                 strides=patch_size, padding='SAME',
                                 kernel_initializer=keras.initializers.LecunNormal(),
                                 bias_initializer=keras.initializers.Zeros()
                                 )

    def call(self, inputs, **kwargs):
        B, H, W, C = inputs.shape

        # we must fix the input image size for vit model
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.projection(inputs)

        # Flatten the spatial Patch embedding of input image into a sequence of embeddings
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [B, self.num_patches, self.embed_dim])  # se14*14 = 196  == sequence length
        return x


class ConcatClassTokenAddPosEmbed(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_patches=196, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        # self.pos_embed = None
        # self.cls_token = None
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def build(self, inputs, **kwargs):
        self.cls_token = self.add_weight(name='cls',
                                         shape=[1, 1, self.embed_dim],
                                         initializer=keras.initializers.Zeros(),
                                         trainable=True,
                                         dtype=tf.float32
                                         )
        self.pos_embed = self.add_weight(name='pos_embed',
                                         shape=[1, self.num_patches + 1, self.embed_dim],
                                         initializer=keras.initializers.RandomNormal(stddev=0.02),
                                         trainable=True,
                                         dtype=tf.float32)

    def call(self, inputs, **kwargs):
        batch_size, _, _ = inputs.shape
        # broadcast cls_token from [1, 1, 768] to Batch size [B, 1, 768]
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)
        x = x + self.pos_embed

        return x


class Attention(keras.layers.Layer):
    # Glorot_uniform draw samples from [-limit, limit] range uniformly
    # Where limit = sqrt(6 / (fan_in + fan_out)), and fan_in is the size of input neurons
    # & fan_out is the size of output neurons
    #
    initializer1 = keras.initializers.GlorotUniform()
    initializer2 = keras.initializers.Zeros()

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0., name=None):
        super(Attention, self).__init__(name=name)

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or tf.math.pow(float(head_dim), -0.5)
        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv',
                                      kernel_initializer=self.initializer1,
                                      bias_initializer=self.initializer2)
        self.attn_drop = Dropout(attn_drop_ratio)

        self.proj = Dense(dim, kernel_initializer=self.initializer1,
                          bias_initializer=self.initializer2)
        self.proj_drop = Dropout(proj_drop_ratio)

    def call(self, inputs, training=None, **kwargs):
        B, N, C = inputs.shape
        out = self.qkv(inputs)
        # reshape: -> [batch_size, num_patches+1, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(out, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attn matrix shape [batch_size, num_heads, num_patches+1, num_patches+1]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        # Softmax won't change the dimension of matrix
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)  # dropout is closed when doing inference

        # shape of attention * value == [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose the matrix to shape = [batch_size, num_patches+1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Here for each attention block the hidden size of inputs == 512
class MLP(keras.layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = keras.initializers.GlorotUniform()
    b_ini = keras.initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        # 512*4 = 2048 but
        self.fc1 = keras.layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                      kernel_initializer=self.k_ini, bias_initializer=self.b_ini, activation='gelu')
        self.fc2 = keras.layers.Dense(in_features, name="Dense_1",
                                      kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = keras.layers.Dropout(drop)

    def call(self, inputs, training=None, **kwargs):
        x = self.fc1(inputs)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


# Definition of Attention Block
class Block(keras.layers.Layer):
    def __init__(self,
                 dim,  # embed_hidden_size
                 num_heads=8,
                 qkv_bias=False,  # Does ViT use less bias initialization
                 qk_scale=None,  # scale the output according to sequence length
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 name=None):
        super(Block, self).__init__(name=name)
        # Applying layer_normalization and skip connection over original inputs
        self.norm1 = LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              name="MultiHeadAttention")
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) if drop_path_ratio > 0. \
            else Activation("linear")
        self.norm2 = LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = MLP(dim, drop=drop_ratio, name="MlpBlock")

    def call(self, inputs, training=None, **kwargs):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class VisionTransformer(keras.Model):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 representation_size=None, num_classes=1000, name="ViT-B/16"):
        super(VisionTransformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.qkv_bias = qkv_bias

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim,
                                                               num_patches=num_patches,
                                                               name="cls_pos")

        self.pos_drop = Dropout(drop_ratio)

        dpr = np.linspace(0., drop_path_ratio, depth)  # stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                             drop_path_ratio=dpr[i], name="encoderblock_{}".format(i))
                       for i in range(depth)]

        self.norm = LayerNormalization(epsilon=1e-6, name="encoder_norm")

        if representation_size:
            self.has_logits = True
            self.pre_logits = Dense(representation_size, activation="tanh", name="pre_logits")
        else:
            self.has_logits = False
            self.pre_logits = Activation("linear")

        self.head = Dense(num_classes, name="head", kernel_initializer=keras.initializers.Zeros())

    def call(self, inputs, training=None, **kwargs):
        # [B, H, W, C] -> [B, num_patches, embed_dim]
        x = self.patch_embed(inputs)  # [B, 196, 768]
        x = self.cls_token_pos_embed(x)  # [B, 176, 768]
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        return x


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-B_16")
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-B_32")
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_16")
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-L_32")
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes,
                              name="ViT-H_14")
    return


if __name__ == '__main__':
    input_img = tf.random.uniform((4, 224, 224, 3))  # random values between 0 to 1
    patch_embed_encoder = PatchEmbed()
    out_img = patch_embed_encoder(input_img)
    add_cls = ConcatClassTokenAddPosEmbed()
    out_img = add_cls(out_img)
    # print(input_img)
    # atten = Attention(dim=768)
    # out_img = atten(out_img)

    block = Block(dim=768)
    out_img = block(out_img)

    print(out_img.shape)
    vit_base_patch16_224_in21k()
    print(111)








