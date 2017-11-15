"""Steer export.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import os
from os.path import join, isfile, isdir
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spiker
from spiker.data import ddd17
from spiker.models import utils

#  def find_best(exp_dir):
#      """find best experiment."""
#      exp_dir = os.path.join(spiker.SPIKER_EXPS+"-run-3", exp_dir)
#      file_list = os.listdir(exp_dir)
#      file_clean_list = []
#      for item in file_list:
#          if ".hdf5" in item:
#              file_clean_list.append(item)
#      file_list = sorted(file_clean_list)
#      return file_list[-1]


def get_prediction(X_test, exp_type, model_base, sensor_type, model_file):
    """Get prediction."""
    model_file_base = exp_type+model_base+sensor_type
    model_path = os.path.join(
        spiker.SPIKER_EXPS+"-run-3", model_file_base,
        model_file)
    print ("[MESSAGE]", model_path)
    model = utils.keras_load_model(model_path)
    prediction = utils.keras_predict_batch(model, X_test, verbose=True)

    return prediction

data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "jul28/rec1501288723-export.hdf5")
frame_cut = [500, 1000]
model_base = "-day-4-"
exp_type = "steering"
sensor_type = ["full", "dvs", "aps"]


load_prediction = os.path.join(
    spiker.SPIKER_EXTRA, "pred"+model_base+"result-run-3")

if os.path.isfile(load_prediction):
    print ("[MESSAGE] Prediction available")
    with open(load_prediction, "r") as f:
        (steer_full, steer_dvs, steer_aps) = pickle.load(f)
        f.close()
else:
    # export ground truth
    test_frames, _ = ddd17.prepare_train_data(data_path,
                                              y_name="steering",
                                              frame_cut=frame_cut)
    test_frames /= 255.
    test_frames -= np.mean(test_frames, keepdims=True)
    num_samples = test_frames.shape[0]
    num_train = int(num_samples*0.7)
    X_test = test_frames[num_train:]
    del test_frames

    # steering full
    steer_full = get_prediction(
        X_test, exp_type, model_base, sensor_type[0],
        "steering-day-4-full-103-0.02.hdf5")
    print ("[MESSAGE] Steering Full")
    # steering dvs 
    steer_dvs = get_prediction(
        X_test[:, :, :, 0][..., np.newaxis],
        exp_type, model_base, sensor_type[1],
        "steering-day-4-dvs-200-0.03.hdf5")
    print ("[MESSAGE] Steering DVS")

    # steering aps
    steer_aps = get_prediction(
        X_test[:, :, :, 1][..., np.newaxis],
        exp_type, model_base, sensor_type[2],
        "steering-day-4-aps-118-0.03.hdf5")
    print ("[MESSAGE] Steering APS")

    del X_test
    save_prediction = os.path.join(
        spiker.SPIKER_EXTRA, "pred"+model_base+"result-run-3")
    with open(save_prediction, "w") as f:
        pickle.dump([steer_full, steer_dvs, steer_aps], f)

origin_data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                                "jul28/rec1501288723.hdf5")
num_samples = 500
frames, steering = ddd17.prepare_train_data(data_path,
                                            target_size=None,
                                            y_name="steering",
                                            frame_cut=frame_cut,
                                            data_portion="test",
                                            data_type="uint8",
                                            num_samples=num_samples)
steering = ddd17.prepare_train_data(data_path,
                                    target_size=None,
                                    y_name="steering",
                                    only_y=True,
                                    frame_cut=frame_cut,
                                    data_portion="test",
                                    data_type="uint8")
steer, steer_time = ddd17.export_data_field(
    origin_data_path, ['steering_wheel_angle'], frame_cut=frame_cut,
    data_portion="test")
steer_time -= steer_time[0]
# in ms
steer_time = steer_time.astype("float32")/1e6
print (steer_time)

idx = 250

fig = plt.figure(figsize=(10, 8))
outer_grid = gridspec.GridSpec(2, 1, wspace=0.1)

# plot frames
frame_grid = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_grid[0, 0],
    hspace=0.1)
aps_frame = plt.Subplot(fig, frame_grid[0])
aps_frame.imshow(frames[idx, :, :, 1], cmap="gray")
aps_frame.axis("off")
aps_frame.set_title("APS Frame")
fig.add_subplot(aps_frame)
dvs_frame = plt.Subplot(fig, frame_grid[1])
dvs_frame.imshow(frames[idx, :, :, 0], cmap="gray")
dvs_frame.axis("off")
dvs_frame.set_title("DVS Frame")
fig.add_subplot(dvs_frame)

# plot steering curve
steering_curve = plt.Subplot(fig, outer_grid[1, 0])
min_steer = np.min(steering*180/np.pi)
max_steer = np.max(steering*180/np.pi)
steering_curve.plot(steer_time, steering*180/np.pi,
                    label="groundtruth",
                    color="#08306b",
                    linestyle="-",
                    linewidth=2)
steering_curve.plot(steer_time, steer_dvs*180/np.pi,
                    label="DVS",
                    color="#3f007d",
                    linestyle="-",
                    linewidth=1)
steering_curve.plot(steer_time, steer_aps*180/np.pi,
                    label="APS",
                    color="#00441b",
                    linestyle="-",
                    linewidth=1)
steering_curve.plot(steer_time, steer_full*180/np.pi,
                    label="DVS+APS",
                    color="#7f2704",
                    linestyle="-",
                    linewidth=1)
steering_curve.plot((steer_time[idx], steer_time[idx]),
                    (min_steer, max_steer), color="black",
                    linestyle="-", linewidth=1)
steering_curve.set_xlim(left=0, right=steer_time[-1])
steering_curve.set_title("Steering Wheel Angle Prediction")
steering_curve.grid(linestyle="-.")
steering_curve.legend(fontsize=10)
steering_curve.set_ylabel("degree")
steering_curve.set_xlabel("time (s)")
fig.add_subplot(steering_curve)

plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                 "vis"+model_base+"result"+".pdf"),
            dpi=600, format="pdf",
            bbox="tight", pad_inches=0.5)
