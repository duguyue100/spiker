"""ResNets.

With only TensorFlow backend.

Follows fchollet's design

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from builtins import range

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2


def resnet_block(input_tensor, kernel_size,
                 filters, stage, block, strides=(2, 2),
                 block_type="identity", bottleneck=True):
    """ResNet Block.

    Parameters
    ----------
    input_tensors : Keras tensor
        input tensor
    kernel_size :
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(0.0001),
                      bias_initializer="zeros",
                      name=conv_name_base+"1")(input_tensor) \
        if block_type == "conv" else input_tensor

    # first transition
    if bottleneck is True:
        x = Conv2D(filters1, (1, 1), strides=strides,
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001),
                   bias_initializer="zeros",
                   name=conv_name_base+"2a")(input_tensor) \
            if block_type == "conv" else \
            Conv2D(filters1, (1, 1),
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001),
                   bias_initializer="zeros",
                   name=conv_name_base+"2a")(input_tensor)
    else:
        x = Conv2D(filters1, kernel_size, strides=strides,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001),
                   bias_initializer="zeros",
                   name=conv_name_base+"2a")(input_tensor) \
            if block_type == "conv" else \
            Conv2D(filters1, kernel_size,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001),
                   bias_initializer="zeros",
                   name=conv_name_base+"2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2a")(x)
    x = Activation("relu")(x)

    # second transition
    x = Conv2D(filters2, kernel_size,
               kernel_initializer="he_normal",
               kernel_regularizer=l2(0.0001),
               bias_initializer="zeros",
               padding="same", name=conv_name_base+"2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2b")(x)
    x = Activation("relu")(x)

    # third transition
    if bottleneck is True:
        x = Conv2D(filters3, (1, 1),
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(0.0001),
                   bias_initializer="zeros",
                   name=conv_name_base+"2c")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2c")(x)

    # merge
    x = add([x, shortcut])
    x = Activation("relu")(x)

    return x


def resnet_builder(model_name, input_shape, batch_size, filter_list,
                   kernel_size, output_dim, stages, blocks, bottleneck=True):
    """Build ResNet."""
    bn_axis = 3

    # prepare input
    img_input = Input(batch_shape=(batch_size,)+input_shape)

    # pre stage
    x = Conv2D(filter_list[0][-1], kernel_size, padding="same",
               kernel_initializer="he_normal",
               kernel_regularizer=l2(0.0001),
               bias_initializer="zeros",
               name="conv1")(img_input)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("relu")(x)

    # first stage
    for block_idx in range(blocks):
        x = resnet_block(x, kernel_size, filter_list[0],
                         1, str(block_idx+1), strides=(1, 1),
                         block_type="identity", bottleneck=bottleneck)

    for stage_idx in range(1, stages):
        # input block
        x = resnet_block(x, kernel_size, filter_list[stage_idx],
                         stage_idx+1, "1", strides=(2, 2),
                         block_type="conv", bottleneck=bottleneck)
        for block_idx in range(1, blocks):
            x = resnet_block(x, kernel_size, filter_list[stage_idx],
                             stage_idx+1, str(block_idx+1), strides=(1, 1),
                             block_type="identity", bottleneck=bottleneck)

    # post stage
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D(data_format="channels_last",
                               name="avg-pool")(x)
    x = Dense(output_dim,
              kernel_initializer="he_normal",
              kernel_regularizer=l2(0.0001),
              bias_initializer="zeros",
              name="output")(x)

    model = Model(img_input, x, name="model_name")

    return model
