"""DDD17 dataset loading test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

import spiker
from spiker.data import ddd17

data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-down-1-export.hdf5")

dvs_frame, aps_frame, steering = ddd17.prepare_train_data(data_path)

print (dvs_frame.shape)
print (aps_frame.shape)
print (steering.shape)

dvs_sample = np.array(dvs_frame[300], dtype=np.float32)
dvs_sample = np.asarray(dvs_sample/np.max(dvs_sample)*256,
                        dtype=np.uint8)

plt.figure()
plt.imshow(dvs_sample, cmap="gray")
#  plt.plot(steering[50:-350])
plt.show()
