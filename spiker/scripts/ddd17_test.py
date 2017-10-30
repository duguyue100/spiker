"""Testing DDD17 utility.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import os

import numpy as np

import spiker
from spiker import data
from spiker.data import ddd17
from spiker import log

# load data
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-down-2.hdf5")


f_in = ddd17.HDF5Stream(file_name, ddd17.EXPORT_DATA_VI.union({'dvs'}))
merged = ddd17.MergedStream(f_in)

time_start = merged.tmin+1e6*0
time_stop = merged.tmax

log.log(time_start)
log.log(time_stop)
log.log((time_stop-time_start)/1e6/60)

# find the start point
merged.search(time_start)

# collect all time
dtypes = {k: float for k in ddd17.EXPORT_DATA.union({'timestamp'})}

# exporting both aps and dvs
dtypes['aps_frame'] = (np.uint8, data.DAVIS346_SHAPE)
dtypes['dvs_frame'] = (np.int16, data.DAVIS346_SHAPE)

outfile = file_name[:-5] + '-export.hdf5'

f_out = ddd17.HDF5(outfile, dtypes, mode='w',
                   chunksize=8, compression='gzip')

current_row = {k: 0 for k in dtypes}

log.log(current_row)
