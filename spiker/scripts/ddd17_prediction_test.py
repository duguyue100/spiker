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
                         "jul09/rec1499657850-export.hdf5")
frame_cut = [500, 500]
frames, steering = ddd17.prepare_train_data(
    data_path, y_name="steering",
    frame_cut=frame_cut)
frames /= 255.
frames -= np.mean(frames, keepdims=True)
num_samples = frames.shape[0]
num_train = int(num_samples*0.7)
X_train = frames[:num_train]
Y_train = steering[:num_train]
X_test = frames[num_train:]
Y_test = steering[num_train:]
del frames, steering

model_name_base = "steering-night-2-"
# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, model_name_base+"full",
    model_name_base+"full-00-0.94.hdf5")
model = utils.keras_load_model(model_path)
aps_dvs_prediction = utils.keras_predict_batch(model, X_test, verbose=True)
x_axis = np.arange(X_test.shape[0])

#  load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, model_name_base+"dvs",
    model_name_base+"dvs-93-0.45.hdf5")
model = utils.keras_load_model(model_path)
dvs_prediction = utils.keras_predict_batch(
    model, X_test[..., 0][..., np.newaxis], verbose=True)

# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, model_name_base+"aps",
    model_name_base+"aps-01-1.03.hdf5")
model = utils.keras_load_model(model_path)
aps_prediction = utils.keras_predict_batch(
    model, X_test[..., 1][..., np.newaxis], verbose=True)

# plot figure
plt.figure(figsize=(10, 4))
plt.plot(x_axis, Y_test, "r", label="ground truth")
plt.plot(x_axis, aps_dvs_prediction, "g", label="DVS+APS")
plt.plot(x_axis, dvs_prediction, "b", label="DVS")
plt.plot(x_axis, aps_prediction, "orange", label="APS")
plt.title("night 2")
plt.xlabel("frames")
plt.ylabel("radian")
plt.legend()
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "night-2-steering.png"),
            dpi=200, format="png")
