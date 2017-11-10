"""Export DDD17 dataset as video.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

import spiker
from spiker.data import ddd17


# load data
data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug02/rec1501705481-export.hdf5")

#  frames, steering = ddd17.prepare_train_data(data_path,
#                                              target_size=None,
#                                              num_samples=1000,
#                                              frame_cut=[0, 2000])
steering = ddd17.prepare_train_data(data_path,
                                    y_name="steering",
                                    only_y=True,
                                    frame_cut=[500, 1000],
                                    num_samples=1000)
x_axis = range(steering.shape[0])
fps = 20
duration = steering.shape[0]/float(fps)
# define a movie clip
fig, ax = plt.subplots()


def make_aps_dvs_frame(t):
    """Make aps and dvs combined frame."""
    idx = int(t*fps)
    ax.plot(x_axis, np.ma.masked_where(x_axis > idx, steering))
    return mplfig_to_npimage(fig)
    #  dvs_frame = np.repeat(frames[idx, :, :, 0][..., np.newaxis], 3, axis=2)
    #  aps_frame = np.repeat(frames[idx, :, :, 1][..., np.newaxis], 3, axis=2)
    #  combined_frame = np.concatenate((dvs_frame, aps_frame), axis=1)
    #  return combined_frame


clip = mpy.VideoClip(make_aps_dvs_frame, duration=duration)

clip.write_videofile(os.path.join(spiker.SPIKER_EXTRA, "test-video.mp4"),
                     fps=fps)
