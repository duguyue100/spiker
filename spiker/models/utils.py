"""Models utilities.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

try:
    from keras.models import load_model
except ImportError:
    print ("[MESSAGE] There is no keras available.")


def load_keras_model(model_file, verbose=False):
    """Load Keras Model.

    TODO: to make the function accept custom classes
    TODO: support different loading mode such as load from json and weights

    Parameters
    ----------
    model_file : string
        absolute path to the pretrained model.
    verbose : bool
        print debug log if necessary
    """
    if not os.path.isfile(model_file):
        raise ValueError("The pretrained model does not exist at %s"
                         % (model_file))

    return load_model(model_file)
