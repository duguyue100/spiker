"""DDD17 Prediction from Pretrained Model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

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
num_samples = frames.shape[0]
num_train = int(num_samples*0.7)
#  X_train = frames[:num_train]
#  Y_train = steering[:num_train]
X_test = frames[num_train:]
Y_test = steering[num_train:]

# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-3-5",
    "resnet-steering-3-5-104-0.01.hdf5")
model = utils.keras_load_model(model_path)

# generate prediction
aps_dvs_prediction = utils.keras_predict_batch(model, X_test, verbose=True)

x_axis = np.arange(X_test.shape[0])

plt.figure(figsize=(10, 4))
plt.plot(x_axis, Y_test, "r", label="ground truth")
plt.plot(x_axis, aps_dvs_prediction, "g", label="predicted")
plt.title("APS and DVS combined.")
plt.xlabel("frames")
plt.ylabel("radius")
plt.legend()
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "aps-dvs-prediction.png"),
            dpi=200, format="png")

#  load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-aps-3-5",
    "resnet-steering-aps-3-5-73-0.01.hdf5")
model = utils.keras_load_model(model_path)
aps_prediction = utils.keras_predict_batch(
    model, X_test[..., 1][..., np.newaxis], verbose=True)

plt.figure(figsize=(10, 4))
plt.plot(x_axis, Y_test, "r", label="ground truth")
plt.plot(x_axis, aps_prediction, "g", label="predicted")
plt.title("APS only.")
plt.xlabel("frames")
plt.ylabel("radius")
plt.legend()
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "aps-prediction.png"),
            dpi=200, format="png")

# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-dvs-3-5",
    "resnet-steering-dvs-3-5-132-0.04.hdf5")
model = utils.keras_load_model(model_path)
aps_prediction = utils.keras_predict_batch(
    model, X_test[..., 0][..., np.newaxis], verbose=True)

plt.figure(figsize=(10, 4))
plt.plot(x_axis, Y_test, "r", label="ground truth")
plt.plot(x_axis, aps_prediction, "g", label="predicted")
plt.title("DVS only.")
plt.xlabel("frames")
plt.ylabel("radius")
plt.legend()
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "dvs-prediction.png"),
            dpi=200, format="png")
