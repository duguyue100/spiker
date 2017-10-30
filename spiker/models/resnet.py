"""ResNets.

With only TensorFlow backend.

Follows fchollet's design

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import add


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
                      name=conv_name_base+"1")(input_tensor) \
        if block_type == "conv" else input_tensor

    # first transition
    if bottleneck is True:
        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base+"2a")(input_tensor) \
            if block_type == "conv" else \
            Conv2D(filters1, (1, 1),
                   name=conv_name_base+"2a")(input_tensor)
    else:
        x = Conv2D(filters1, kernel_size, strides=strides,
                   name=conv_name_base+"2a")(input_tensor) \
            if block_type == "conv" else \
            Conv2D(filters1, kernel_size,
                   name=conv_name_base+"2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2a")(x)
    x = Activation("relu")(x)

    # second transition
    x = Conv2D(filters2, kernel_size,
               padding="same", name=conv_name_base+"2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2b")(x)
    x = Activation("relu")(x)

    # third transition
    if bottleneck is True:
        x = Conv2D(filters3, (1, 1), name=conv_name_base+"2c")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base+"2c")(x)

    # merge
    x = add([x, shortcut])
    x = Activation("relu")(x)

    return x
