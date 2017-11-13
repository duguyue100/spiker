"""Export DDD17 dataset as video.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

import spiker
from spiker.data import ddd17
from spiker.models import utils


def find_best(exp_dir):
    """find best experiment."""
    exp_dir = os.path.join(spiker.SPIKER_EXPS, exp_dir)
    file_list = os.listdir(exp_dir)
    file_clean_list = []
    for item in file_list:
        if ".hdf5" in item:
            file_clean_list.append(item)
    file_list = sorted(file_clean_list)
    return file_list[-1]


def get_prediction(X_test, exp_type, model_base, sensor_type):
    """Get prediction."""
    model_file_base = exp_type+model_base+sensor_type
    model_path = os.path.join(
        spiker.SPIKER_EXPS, model_file_base,
        find_best(model_file_base))
    print ("[MESSAGE]", model_path)
    model = utils.keras_load_model(model_path)
    prediction = utils.keras_predict_batch(model, X_test, verbose=True)

    return prediction

# load data
data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "jul09/rec1499656391-export.hdf5")
frame_cut = [2000, 4000]
# load model
model_base = "-night-1-"
exp_type = ["steering", "accel", "brake"]
sensor_type = ["full", "dvs", "aps"]

# get prediction
load_prediction = os.path.join(
    spiker.SPIKER_EXTRA, "pred"+model_base+"result")

if os.path.isfile(load_prediction):
    print ("[MESSAGE] Prediction available")
    with open(load_prediction, "r") as f:
        (steer_full, steer_dvs, steer_aps,
         accel_full, accel_dvs, accel_aps,
         brake_full, brake_dvs, brake_aps) = pickle.load(f)
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
        X_test, exp_type[0], model_base, sensor_type[0])
    print ("[MESSAGE] Steering Full")
    # steering dvs 
    steer_dvs = get_prediction(
        X_test[:, :, :, 0][..., np.newaxis],
        exp_type[0], model_base, sensor_type[1])
    print ("[MESSAGE] Steering DVS")
    # steering aps
    steer_aps = get_prediction(
        X_test[:, :, :, 1][..., np.newaxis],
        exp_type[0], model_base, sensor_type[2])
    print ("[MESSAGE] Steering APS")
    # accel full
    accel_full = get_prediction(
        X_test, exp_type[1], model_base, sensor_type[0])
    print ("[MESSAGE] Accel Full")
    # accel dvs 
    accel_dvs = get_prediction(
        X_test[:, :, :, 0][..., np.newaxis],
        exp_type[1], model_base, sensor_type[1])
    print ("[MESSAGE] Accel DVS")
    # accel aps
    accel_aps = get_prediction(
        X_test[:, :, :, 1][..., np.newaxis],
        exp_type[1], model_base, sensor_type[2])
    print ("[MESSAGE] Accel APS")
    # brake full
    brake_full = get_prediction(
        X_test, exp_type[2], model_base, sensor_type[0])
    print ("[MESSAGE] Brake Full")
    # brake dvs 
    brake_dvs = get_prediction(
        X_test[:, :, :, 0][..., np.newaxis],
        exp_type[2], model_base, sensor_type[1])
    print ("[MESSAGE] Brake DVS")
    # brake aps
    brake_aps = get_prediction(
        X_test[:, :, :, 1][..., np.newaxis],
        exp_type[2], model_base, sensor_type[2])
    print ("[MESSAGE] Brake APS")

    del X_test

    # save prediction for future use.
    save_prediction = os.path.join(
        spiker.SPIKER_EXTRA, "pred"+model_base+"result")
    with open(save_prediction, "w") as f:
        pickle.dump([steer_full, steer_dvs, steer_aps,
                     accel_full, accel_dvs, accel_aps,
                     brake_full, brake_dvs, brake_aps], f)
        f.close()

# load visualization data
num_samples = 500
frames, steering = ddd17.prepare_train_data(data_path,
                                            target_size=None,
                                            y_name="steering",
                                            frame_cut=frame_cut,
                                            data_portion="test",
                                            data_type="uint8",
                                            num_samples=num_samples)
accel = ddd17.prepare_train_data(data_path,
                                 y_name="accel",
                                 only_y=True,
                                 frame_cut=frame_cut,
                                 data_portion="test",
                                 num_samples=num_samples)
brake = ddd17.prepare_train_data(data_path,
                                 y_name="brake",
                                 only_y=True,
                                 frame_cut=frame_cut,
                                 data_portion="test",
                                 num_samples=num_samples)
x_axis = np.arange(steering.shape[0])

# video properties
fps = 20
duration = steering.shape[0]/float(fps)


def make_aps_dvs_frame(t):
    """Make aps and dvs combined frame."""
    # identify frame name
    idx = int(t*fps)
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
    curve_grid = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_grid[1, 0],
        hspace=0.35)
    steering_curve = plt.Subplot(fig, curve_grid[0, 0])
    steering_curve.plot(x_axis[:idx], steering[:idx]*180/np.pi,
                        label="groundtruth",
                        color="#08306b",
                        linestyle="-",
                        linewidth=2)
    steering_curve.plot(x_axis[:idx], steer_full[:idx]*180/np.pi,
                        label="DVS+APS",
                        color="#7f2704",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(x_axis[:idx], steer_dvs[:idx]*180/np.pi,
                        label="DVS",
                        color="#3f007d",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(x_axis[:idx], steer_aps[:idx]*180/np.pi,
                        label="APS",
                        color="#00441b",
                        linestyle="-",
                        linewidth=1)
    steering_curve.set_xlim(left=1, right=steering.shape[0])
    steering_curve.set_xticks([])
    steering_curve.set_title("Steering Wheel Angle Prediction")
    steering_curve.grid(linestyle="-.")
    steering_curve.legend(bbox_to_anchor=(1, 0.5), fontsize=10)
    steering_curve.set_ylabel("degree")
    fig.add_subplot(steering_curve)

    # plot accel curve
    accel_curve = plt.Subplot(fig, curve_grid[1, 0])
    accel_curve.plot(x_axis[:idx], accel[:idx]*100,
                     label="groundtruth",
                     color="#08306b",
                     linestyle="-",
                     linewidth=2)
    accel_curve.plot(x_axis[:idx], accel_full[:idx]*100,
                     label="DVS+APS",
                     color="#7f2704",
                     linestyle="-",
                     linewidth=1)
    accel_curve.plot(x_axis[:idx], accel_dvs[:idx]*100,
                     label="DVS",
                     color="#3f007d",
                     linestyle="-",
                     linewidth=1)
    accel_curve.plot(x_axis[:idx], accel_aps[:idx]*100,
                     label="APS",
                     color="#00441b",
                     linestyle="-",
                     linewidth=1)
    accel_curve.set_xlim(left=1, right=accel.shape[0])
    accel_curve.set_xticks([])
    accel_curve.set_title("Accelerator Pedal Position Prediction")
    accel_curve.grid(linestyle="-.")
    accel_curve.legend(bbox_to_anchor=(1, 0.5), fontsize=10)
    accel_curve.set_ylabel("pressure (%)")
    fig.add_subplot(accel_curve)

    # plot brake curve
    brake_curve = plt.Subplot(fig, curve_grid[2, 0])
    brake_curve.plot(x_axis[:idx], brake[:idx]*100,
                     label="groundtruth",
                     color="#08306b",
                     linestyle=" ",
                     marker="o",
                     markersize=4)
    brake_curve.plot(x_axis[:idx], brake_full[:idx],
                     label="DVS+APS",
                     color="#7f2704",
                     linestyle="-",
                     linewidth=2)
    brake_curve.plot(x_axis[:idx], brake_dvs[:idx],
                     label="DVS",
                     color="#3f007d",
                     linestyle="-",
                     linewidth=2)
    brake_curve.plot(x_axis[:idx], brake_aps[:idx],
                     label="APS",
                     color="#00441b",
                     linestyle="-",
                     linewidth=2)
    brake_curve.set_xlim(left=1, right=brake.shape[0])
    brake_curve.set_yticks([0, 100])
    brake_curve.set_yticklabels(["OFF", "ON"])
    brake_curve.set_title("Brake Pedal Position Prediction")
    brake_curve.grid(linestyle="-.")
    brake_curve.legend(bbox_to_anchor=(1, 0.5), fontsize=10)
    brake_curve.set_xlabel("frame")
    brake_curve.set_ylabel("ON/OFF")
    fig.add_subplot(brake_curve)

    # form data array
    fig.canvas.draw()
    data_buffer = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data_buffer = data_buffer.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))

    return data_buffer

clip = mpy.VideoClip(make_aps_dvs_frame, duration=duration)

clip.write_videofile(os.path.join(spiker.SPIKER_EXTRA,
                     "benchmark"+model_base+"video.mp4"),
                     fps=fps)
