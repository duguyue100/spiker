"""HDF5 Test.

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

logger = log.get_logger("hdf5-test", log.INFO)

hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_test.hdf5")

dataset = h5py.File(hdf5_path, "r")

logger.info(dataset["aps/aps_data"].shape)
logger.info(dataset["dvs/event_loc"].shape)
logger.info(dataset["dvs/event_ts"].shape)
logger.info(dataset["dvs/event_pol"].shape)

aps_time = dataset["aps/aps_ts"][()]
pwm_time = dataset["extra/pwm/pwm_ts"][()]
dvs_time = dataset["dvs/event_ts"][19]

logger.info(aps_time[0])
logger.info(pwm_time[0])

time_shift = aps_time[0]

aps_time -= time_shift
aps_y = np.ones((aps_time.shape[0],))
pwm_time -= time_shift
pwm_y = np.ones((pwm_time.shape[0],))*1.1

plt.figure()
#  plt.plot(aps_time, aps_y, "go")
plt.plot(aps_time, aps_y, "go", pwm_time, pwm_y, "ro")
plt.show()

dataset.close()
