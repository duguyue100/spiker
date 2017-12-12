"""Models utilities.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

import numpy as np

from spiker import log

logger = log.get_logger("models-utils", log.DEBUG)

try:
    from keras.models import load_model
except ImportError:
    logger.critical("There is no keras available.")


def keras_load_model(model_file, verbose=False):
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


def keras_predict_batch(model, data, batch_size=64, verbose=False):
    """Predict result in mini-batches for Keras model.

    Parameters
    ----------
    model : a Keras Model
    data : numpy.ndarray
        Numpy array that has data, usually has tensor shape such as NHWC
    batch_size: int
        number of samples evaluated per batch
    verbose : bool
        optional debug message
    """
    num_test = data.shape[0]
    Y_predict = np.array([])

    for batch in xrange(num_test // batch_size):
        y_predict = model.predict(data[batch*batch_size:(batch+1)*batch_size])
        Y_predict = y_predict if Y_predict.size == 0 else \
            np.vstack((Y_predict, y_predict))
        if verbose is True:
            logger.info("Processed %d/%d, size: %d"
                        % (batch+1, num_test//batch_size, Y_predict.shape[0]))
    # only do one more pass if there is reminder
    if num_test % batch_size != 0:
        y_predict = model.predict(data[-(num_test % batch_size):])
        Y_predict = np.vstack((Y_predict, y_predict))

    return Y_predict


def parse_csv_log(csv_log_path, verbose=False):
    """Parse CSV Log.

    Assume only 5 columns.
    NOTICE: assume the file is from Keras csv logger.

    Parameters
    ----------
    csv_log_path : string
        absolute path to the csv log
    verbose : boo
        optional debug message
    """
    if not os.path.isfile(csv_log_path):
        raise ValueError("The CSV Log does not exist at %s" % (csv_log_path))

    log_data = np.genfromtxt(
        csv_log_path, dtype="float32", delimiter=",", names=True,
        usecols=(1, 2, 3, 4))

    if verbose is True:
        logger.info("Available fields are:", log_data.dtype.names)

    return log_data
