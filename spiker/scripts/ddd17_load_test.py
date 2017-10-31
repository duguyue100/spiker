"""DDD17 dataset loading test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import rotate

import spiker
from spiker.data import ddd17

data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-down-1-export.hdf5")

frames, steering = ddd17.prepare_train_data(data_path)

print (frames.shape)
print (steering.shape)

dvs_mean = frames[..., 0].mean(axis=(1, 2))

#  dvs_temp = dvs_frame[6400]
#  dvs_temp = (dvs_temp+127).astype("float32").astype("uint8")
#  dvs_temp = rotate(dvs_temp, angle=180)

plt.figure()
#  plt.imshow(dvs_temp, cmap="gray")
#  plt.plot(steering[50:-350])
plt.plot(dvs_mean[50:-350])
plt.show()