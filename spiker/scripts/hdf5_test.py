"""HDF5 Test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
from builtins import range
import os

import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

import spiker
from spiker import log

logger = log.get_logger("hdf5-test", log.INFO)

hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_exported.hdf5")

dataset = h5py.File(hdf5_path, "r")

logger.info(dataset["aps"].shape)
logger.info(dataset["dvs"].shape)
logger.info(dataset["pwm"].shape)

pwm_data = dataset["pwm"][()]

plt.figure()
# steering data
plt.plot(pwm_data[:, 0])
plt.show()

for frame_id in range(dataset["aps"].shape[0]):
    cv2.imshow("aps", dataset["aps"][frame_id][()])
    cv2.imshow("dvs", dataset["dvs"][frame_id][()]/float(8*2))

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

dataset.close()
