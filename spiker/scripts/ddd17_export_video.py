"""Export DDD17 dataset as video.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle

import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import moviepy.editor as mpy

import spiker
from spiker.data import ddd17
from spiker.models import utils

CV_AA = cv2.LINE_AA if int(cv2.__version__[0]) > 2 else cv2.CV_AA


def plot_steering_wheel(img, steer_angles, colors=[(8, 48, 107)],
                        thickness=[2]):
    """draw angles based on a list of predictions."""
    c, r = (173, 130), 65  # center, radius
    for angle_idx in xrange(len(steer_angles)):
        a = steer_angles[angle_idx]
        a_rad = + a / 180. * np.pi + np.pi / 2
        a_rad = np.pi-a_rad
        t = (c[0] + int(np.cos(a_rad) * r), c[1] - int(np.sin(a_rad) * r))
        cv2.line(img, c, t, colors[angle_idx],
                 thickness[angle_idx], CV_AA)
    cv2.circle(img, c, r, (0, 0, 0), 1, CV_AA)
    # the label
    cv2.line(img, (c[0]-r+5, c[1]), (c[0]-r, c[1]), (0, 0, 0), 1, CV_AA)
    cv2.line(img, (c[0]+r-5, c[1]), (c[0]+r, c[1]), (0, 0, 0), 1, CV_AA)
    cv2.line(img, (c[0], c[1]-r+5), (c[0], c[1]-r), (0, 0, 0), 1, CV_AA)
    cv2.line(img, (c[0], c[1]+r-5), (c[0], c[1]+r), (0, 0, 0), 1, CV_AA)
    cv2.putText(img, 'gt: %0.1f deg' % steer_angles[0], (
        c[0]-35, c[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
        1, CV_AA)
    return img


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


# construct experiment cuts
exp_names = {
    "jul09/rec1499656391-export.hdf5": [2000, 4000],
    "jul09/rec1499657850-export.hdf5": [500, 800],
    "aug01/rec1501649676-export.hdf5": [500, 500],
    "aug01/rec1501650719-export.hdf5": [500, 500],
    "aug05/rec1501994881-export.hdf5": [200, 800],
    "aug09/rec1502336427-export.hdf5": [100, 400],
    "aug09/rec1502337436-export.hdf5": [100, 400],
    "jul16/rec1500220388-export.hdf5": [500, 200],
    "jul18/rec1500383971-export.hdf5": [500, 1000],
    "jul18/rec1500402142-export.hdf5": [200, 2000],
    "jul28/rec1501288723-export.hdf5": [200, 1000],
    "jul29/rec1501349894-export.hdf5": [200, 1500],
    "aug01/rec1501614399-export.hdf5": [200, 800],
    "aug08/rec1502241196-export.hdf5": [500, 1000],
    "aug15/rec1502825681-export.hdf5": [500, 1700]
}

# construct experiment names
exp_des = {
    "jul09/rec1499656391-export.hdf5": "night-1",
    "jul09/rec1499657850-export.hdf5": "night-2",
    "aug01/rec1501649676-export.hdf5": "night-3",
    "aug01/rec1501650719-export.hdf5": "night-4",
    "aug05/rec1501994881-export.hdf5": "night-5",
    "aug09/rec1502336427-export.hdf5": "night-6",
    "aug09/rec1502337436-export.hdf5": "night-7",
    "jul16/rec1500220388-export.hdf5": "day-1",
    "jul18/rec1500383971-export.hdf5": "day-2",
    "jul18/rec1500402142-export.hdf5": "day-3",
    "jul28/rec1501288723-export.hdf5": "day-4",
    "jul29/rec1501349894-export.hdf5": "day-5",
    "aug01/rec1501614399-export.hdf5": "day-6",
    "aug08/rec1502241196-export.hdf5": "day-7",
    "aug15/rec1502825681-export.hdf5": "day-8"
}

for exp in exp_des:
    exp_id = exp_des[exp]
    # load data
    data_path = os.path.join(
        spiker.SPIKER_DATA, "ddd17", exp)
    origin_data_path = os.path.join(
        spiker.SPIKER_DATA, "ddd17",
        exp[:-12]+".hdf5")
    frame_cut = exp_names[exp]
    # frame model base names
    model_base = "-"+exp_id+"-"
    sensor_type = ["full", "dvs", "aps"]

    print ("[MESSAGE] Data path:", data_path)
    print ("[MESSAGE] Original path:", origin_data_path)
    print ("[MESSAGE] Frame cut:", frame_cut)

    num_samples = 1000
    # load ground truth
    frames, steering = ddd17.prepare_train_data(data_path,
                                                target_size=None,
                                                y_name="steering",
                                                frame_cut=frame_cut,
                                                data_portion="test",
                                                data_type="uint8",
                                                num_samples=1000)
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
    steering = steering[:num_samples]
    steer_time = steer_time[:num_samples]
    steer_time -= steer_time[0]
    # in ms
    steer_time = steer_time.astype("float32")/1e6
    print (steer_time)

    # load prediction
    res_path = os.path.join(
        spiker.SPIKER_EXTRA, "exported-results",
        "steering"+model_base+"run-1.pkl")
    with open(res_path, "r") as f:
        run_1 = pickle.load(f)
        f.close()
    res_path = os.path.join(
        spiker.SPIKER_EXTRA, "exported-results",
        "steering"+model_base+"run-2.pkl")
    with open(res_path, "r") as f:
        run_2 = pickle.load(f)
        f.close()
    res_path = os.path.join(
        spiker.SPIKER_EXTRA, "exported-results",
        "steering"+model_base+"run-3.pkl")
    with open(res_path, "r") as f:
        run_3 = pickle.load(f)
        f.close()
    res_path = os.path.join(
        spiker.SPIKER_EXTRA, "exported-results",
        "steering"+model_base+"run-4.pkl")
    with open(res_path, "r") as f:
        run_4 = pickle.load(f)
        f.close()

    # calculating mean and difference
    full_res = np.hstack(
        (run_1[0], run_2[0], run_3[0], run_4[0])).T
    full_mean_res = np.mean(full_res, axis=0)*180.0/np.pi
    full_mean_res = full_mean_res[:num_samples]
    full_std_res = np.std(full_res, axis=0)*180.0/np.pi
    full_std_res = full_std_res[:num_samples]
    dvs_res = np.hstack(
        (run_1[1], run_2[1], run_3[1], run_3[1])).T
    dvs_mean_res = np.mean(dvs_res, axis=0)*180.0/np.pi
    dvs_mean_res = dvs_mean_res[:num_samples]
    dvs_std_res = np.std(dvs_res, axis=0)*180.0/np.pi
    dvs_std_res = dvs_std_res[:num_samples]
    aps_res = np.hstack(
        (run_1[2], run_2[2], run_3[2], run_3[2])).T
    aps_mean_res = np.mean(aps_res, axis=0)*180.0/np.pi
    aps_mean_res = aps_mean_res[:num_samples]
    aps_std_res = np.std(aps_res, axis=0)*180.0/np.pi
    aps_std_res = aps_std_res[:num_samples]

    # video properties
    fps = 20
    duration = steering.shape[0]/float(fps)
    #  duration = 200/float(fps)

    def make_aps_dvs_frame(t):
        """Make aps and dvs combined frame."""
        # identify frame name
        idx = int(t*fps)
        # producing figures
        fig = plt.figure(figsize=(10, 8))
        outer_grid = gridspec.GridSpec(2, 1, wspace=0.1)

        # plot frames
        frame_grid = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_grid[0, 0],
            hspace=0.1)
        # plot aps frame
        aps_frame = plt.Subplot(fig, frame_grid[0])
        aps_frame_plot = frames[idx, :, :, 1][..., np.newaxis]
        aps_frame_plot = aps_frame_plot.repeat(3, axis=2)
        aps_frame_plot = plot_steering_wheel(
            aps_frame_plot,
            [steering[idx]*180/np.pi, full_mean_res[idx],
             aps_mean_res[idx]],
            [(107, 48, 8), (4, 39, 127), (27, 68, 0)],
            [2, 1, 1])
        aps_frame.imshow(aps_frame_plot)
        aps_frame.axis("off")
        aps_frame.set_title("APS Frame")
        fig.add_subplot(aps_frame)

        # plot dvs frame
        dvs_frame = plt.Subplot(fig, frame_grid[1])
        dvs_frame_plot = frames[idx, :, :, 0][..., np.newaxis]
        dvs_frame_plot = dvs_frame_plot.repeat(3, axis=2)
        dvs_frame_plot = plot_steering_wheel(
            dvs_frame_plot,
            [steering[idx]*180/np.pi, full_mean_res[idx],
             dvs_mean_res[idx]],
            [(107, 48, 8), (4, 39, 127), (125, 0, 63)],
            [2, 1, 1])
        dvs_frame.imshow(dvs_frame_plot)
        dvs_frame.axis("off")
        dvs_frame.set_title("DVS Frame")
        fig.add_subplot(dvs_frame)

        # plot steering curve
        steering_curve = plt.Subplot(fig, outer_grid[1, 0])
        min_steer = np.min(steering*180/np.pi)
        max_steer = np.max(steering*180/np.pi)
        steering_curve.plot(steer_time, dvs_mean_res,
                            label="DVS",
                            color="#3f007d",
                            linestyle="-",
                            linewidth=1)
        steering_curve.fill_between(
            steer_time, dvs_mean_res+dvs_std_res,
            dvs_mean_res-dvs_std_res, facecolor="#3f007d",
            alpha=0.3)
        steering_curve.plot(steer_time, aps_mean_res,
                            label="APS",
                            color="#00441b",
                            linestyle="-",
                            linewidth=1)
        steering_curve.fill_between(
            steer_time, aps_mean_res+aps_std_res,
            aps_mean_res-aps_std_res, facecolor="#00441b",
            alpha=0.3)
        steering_curve.plot(steer_time, full_mean_res,
                            label="DVS+APS",
                            color="#7f2704",
                            linestyle="-",
                            linewidth=1)
        steering_curve.fill_between(
            steer_time, full_mean_res+full_std_res,
            full_mean_res-full_std_res, facecolor="#7f2704",
            alpha=0.3)
        steering_curve.plot(steer_time, steering*180/np.pi,
                            label="groundtruth",
                            color="#08306b",
                            linestyle="-",
                            linewidth=2)
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

        outer_grid.tight_layout(fig)

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
