"""Testing DDD17 utility.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import os
import Queue
import time
from copy import deepcopy

import numpy as np

import spiker
from spiker import data
from spiker.data import ddd17
from spiker import log

# load data
#  file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
#                           "highway-down-2.hdf5")
#  file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
#                           "highway-down-1.hdf5")
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-up-1.hdf5")
#  file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
#                           "highway-up-2.hdf5")

# CVPR data
# Night 1
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "jul09/rec1499656391.hdf5")
# Night 2
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "jul09/rec1499657850.hdf5")
# Night 3
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug01/rec1501649676.hdf5")
# Night 4
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug01/rec1501650719.hdf5")
# Night 5
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug05/rec1501994881.hdf5")
# Night 6
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug09/rec1502336427.hdf5")
# Night 7
file_name = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "aug09/rec1502337436.hdf")
# Day 1

# Day 2

# Day 3

# Day 4

# Day 5

# Day 6

# Day 7

# Day 8

binsize = 0.1
fixed_dt = binsize > 0
clip_value = 8

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

# current sample
current_row = {k: 0 for k in dtypes}
current_row['aps_frame'] = np.zeros(data.DAVIS346_SHAPE, dtype=np.uint8)
current_row['dvs_frame'] = np.zeros(data.DAVIS346_SHAPE, dtype=np.int16)

sys_ts, t_pre, t_offset, ev_count, pbar_next = 0, 0, 0, 0, 0

while merged.has_data and sys_ts <= time_stop*1e-6:
    try:
        sys_ts, d = merged.get()
    except Queue.Empty:
        # wait for queue to fill up
        time.sleep(0.01)
        continue
    if not d:
        # skip unused data
        continue
    if d['etype'] == 'special_event':
        ddd17.unpack_data(d)
        if any(d['data'] == 0):
            # this is a timestamp reset
            print('ts reset detected, setting offset',
                  current_row['timestamp'])
            t_offset += current_row['timestamp']
            # NOTE the timestamp of this special event is not meaningful
            continue
    if d['etype'] in ddd17.EXPORT_DATA_VI:
        current_row[d['etype']] = d['data']
        continue
    if t_pre == 0 and d['etype'] in ['frame_event', 'polarity_event']:
        print('resetting t_pre (first %s)' % d['etype'])
        t_pre = d['timestamp'] + t_offset
    if d['etype'] == 'frame_event':
        if fixed_dt:
            while t_pre + binsize < d['timestamp'] + t_offset:
                # aps frame is not in current bin -> save and proceed
                f_out.save(deepcopy(current_row))
                current_row['dvs_frame'][:, :] = 0
                current_row['timestamp'] = t_pre
                t_pre += binsize
        else:
            current_row['timestamp'] = d['timestamp'] + t_offset
        current_row['aps_frame'] = ddd17.filter_frame(ddd17.unpack_data(d))
        # current_row['timestamp'] = t_pre
        # JB: I don't see why the previous line should make sense
        continue
    if d['etype'] == 'polarity_event':
        ddd17.unpack_data(d)
        times = d['data'][:, 0] * 1e-6 + t_offset
        num_evts = d['data'].shape[0]
        print ("Number of events in this frame: %d"
               % (num_evts))
        offset = 0
        if fixed_dt:
            # fixed time interval bin mode
            num_samples = int(np.ceil((times[-1] - t_pre) / binsize))
            for _ in xrange(num_samples):
                # take n events
                n = (times[offset:] < t_pre + binsize).sum()
                sel = slice(offset, offset + n)
                current_row['dvs_frame'] += \
                    ddd17.raster_evts(d['data'][sel], clip_value=1)
                #  current_row['dvs_frame'] += \
                #      ddd17.raster_evts_new(d['data'][sel])
                offset += n
                # save if we're in the middle of a packet, otherwise
                # wait for more data
                if sel.stop < num_evts:
                    current_row['timestamp'] = t_pre
                    f_out.save(deepcopy(current_row))
                    current_row['dvs_frame'][:, :] = 0
                    t_pre += binsize
            np.clip(current_row['dvs_frame'], -clip_value, clip_value,
                    out=current_row['dvs_frame'])
        else:
            # fixed event count mode
            num_samples = np.ceil(-float(num_evts + ev_count)/binsize)
            for _ in xrange(int(num_samples)):
                n = min(int(-binsize - ev_count), num_evts - offset)
                sel = slice(offset, offset + n)
                current_row['dvs_frame'] += ddd17.raster_evts(d['data'][sel])
                if sel.stop > sel.start:
                    current_row['timestamp'] = times[sel].mean()
                offset += n
                ev_count += n
                if ev_count == -binsize:
                    f_out.save(deepcopy(current_row))
                    current_row['dvs_frame'][:, :] = 0
                    ev_count = 0
            np.clip(current_row['dvs_frame'], -clip_value, clip_value,
                    out=current_row['dvs_frame'])

print('[DEBUG] sys_ts/time_stop', sys_ts, time_stop*1e-6)
merged.exit.set()
f_out.exit.set()
f_out.join()
print('[DEBUG] output done')
while not merged.done.is_set():
    print('[DEBUG] waiting for merger')
    time.sleep(1)
print('[DEBUG] merger done')
f_in.join()
print('[DEBUG] stream joined')
merged.join()
print('[DEBUG] merger joined')
filesize = os.path.getsize(outfile)
print('Finished.  Wrote {:.1f}MiB to {}.'.format(filesize/1024**2, outfile))

time.sleep(1)
os._exit(0)
