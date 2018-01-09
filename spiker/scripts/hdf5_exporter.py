"""HDF5 Exporter.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
from builtins import range
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

import spiker
from spiker import log


def determine_aps_cut(aps_time, pwm_time):
    """Determine data start point.

    This function finds the first and last valid pwm and aps
    signal based on time.
    """
    if aps_time[0] <= pwm_time[0]:
        # pwm arrives after aps
        aps_head = np.nonzero(aps_time > pwm_time[0])[0][0]-1
        pwm_head = 0
    elif aps_time[0] > pwm_time[0]:
        # pwm arrives before aps
        aps_head = 0
        pwm_head = np.nonzero(pwm_time > aps_time[0])[0][0]

    if aps_time[-1] <= pwm_time[-1]:
        # pwm finish after
        aps_tail = aps_time.shape[0]
        pwm_tail = np.nonzero(pwm_time > aps_time[-1])[0][0]-1
    elif aps_time[-1] > pwm_time[-1]:
        # pwm finish before
        aps_tail = np.nonzero(aps_time > pwm_time[-1])[0][0]
        pwm_tail = pwm_time.shape[0]

    return aps_head, pwm_head, aps_tail, pwm_tail


logger = log.get_logger("hdf5-exporter", log.INFO)

dvs_bin_size = 100  # ms

hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_test.hdf5")

dataset = h5py.File(hdf5_path, "r")

# basic stats
aps_time = dataset["aps/aps_ts"][()]
pwm_time = dataset["extra/pwm/pwm_ts"][()]

aps_head, pwm_head, aps_tail, pwm_tail = determine_aps_cut(aps_time, pwm_time)

aps_data = dataset["aps/aps_data"][aps_head:aps_tail][()]
aps_time = aps_time[aps_head:aps_tail]
num_imgs = aps_data.shape[0]
pwm_data = dataset["extra/pwm/pwm_data"][pwm_head:pwm_tail][()]
pwm_time = pwm_time[pwm_head:pwm_tail]
num_cmds = pwm_data.shape[0]

logger.info(num_imgs)
logger.info(num_cmds)

# since pwm is sampled around 10Hz in a very stable rate
# use as a sync signal

num_samples = max(num_cmds, num_imgs)

aps_data_new = np.zeros((num_samples, aps_data.shape[1], aps_data.shape[2]),
                        dtype=np.uint8)
dvs_data_new = np.zeros((num_samples, aps_data.shape[1], aps_data.shape[2]),
                        dtype=np.uint8)
pwm_data_new = np.zeros((num_samples, pwm_data.shape[1]), dtype=np.float32)
pwm_data_new[0] = pwm_data[0]
aps_data_new[0] = aps_data[0]
for cmd_idx in range(1, num_cmds):
    # current command time
    curr_cmd_time = pwm_time[cmd_idx]
    # previous command time
    prev_cmd_time = pwm_time[cmd_idx-1]

    # find all frames between two cmd
    frame_idxs = np.nonzero(
        (prev_cmd_time < aps_time)*(aps_time < curr_cmd_time))[0]
    # assign data
    if frame_idxs.shape[0] == 0:
        # no data, frame rate too low
        pass
    else:
        # there is frame(s) between two command
        for idx in range(frame_idxs.shape[0]-1):
            pwm_data_new[frame_idxs[idx]] = \
                (pwm_data[cmd_idx]+pwm_data[cmd_idx-1])/2
            aps_data_new[frame_idxs[idx]] = aps_data[frame_idxs[idx]]
            # make dvs between current and next frame
            #  bin_size = min(
            #      (aps_time[frame_idxs[idx]+1]-aps_time[frame_idxs[idx]])/1e3,
            #      dvs_bin_size)
            # find event range and bind the frame

        # assign last frame to the steering
        pwm_data_new[frame_idxs[-1]] = pwm_data[cmd_idx]
        aps_data_new[frame_idxs[-1]] = aps_data[frame_idxs[-1]]
        # make dvs between current and next frame
        #  bin_size = min(
        #      (aps_time[frame_idxs[-1]+1]-aps_time[frame_idxs[-1]])/1e3,
        #      dvs_bin_size)
        # find event range and bind the frame

dataset.close()

#  write to new file
hdf5_path_new = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_test_exported.hdf5")

dataset = h5py.File(hdf5_path_new, "w")

dataset.create_dataset("aps", data=aps_data_new, dtype="uint8")
dataset.create_dataset("dvs", data=dvs_data_new, dtype="uint8")
dataset.create_dataset("pwm", data=pwm_data_new, dtype="float32")

dataset.close()
