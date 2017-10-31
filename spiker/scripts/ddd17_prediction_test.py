"""DDD17 Prediction from Pretrained Model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np

import spiker
from spiker.models import utils
from spiker.data import ddd17

# load and process data
data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-down-1-export.hdf5")
frames, steering = ddd17.prepare_train_data(data_path)
frames = frames[50:-350]/255.
frames -= np.mean(frames, keepdims=True)
steering = steering[50:-350]

# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-3-5")
model = utils.load_keras_model(model_path)

# generate prediction
