"""Export Frame and Event rate for recording.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import cPickle as pickle

import os
import Queue
import time

import numpy as np

import spiker
from spiker import data
from spiker.data import ddd17
from spiker import log

# import csv file
csv_file_path = os.path.join(spiker.SPIKER_DATA, "ddd17", "DDD17plus.csv")
data_path_base = "/run/user/1000/gvfs/smb-share:server=" \
    "sensors-nas.lan.ini.uzh.ch,share=resiliosync/" \
    "DDD17-fordfocus/fordfocus"
# read csv file
recording_list = np.loadtxt(csv_file_path, dtype=str, delimiter=",")[:, 1]

for data_item in recording_list:
    record_path = os.path.join(data_path_base, data_item)
    out_file = os.path.join(spiker.SPIKER_EXTRA, "exported-rate",
                            data_item[6:-5]+".pkl")
    if os.path.isfile(out_file):
        print ("[MESSAGE] Skipping", out_file)
        continue

    f_in = ddd17.HDF5Stream(record_path, {'dvs'})
    merged = ddd17.MergedStream(f_in)

    time_start = merged.tmin+1e6*0
    time_stop = merged.tmax

    # find the start point
    merged.search(time_start)

    # current sample
    sys_ts = 0
    num_aps_frame = 0
    num_dvs_events = 0

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
        if d['etype'] == 'frame_event':
            num_aps_frame += 1
            continue
        if d['etype'] == 'polarity_event':
            ddd17.unpack_data(d)
            num_evts = d['data'].shape[0]
            num_dvs_events += num_evts
            print ("Number of events in this frame: %d"
                   % (num_evts))

    # save statistics
    with open(out_file, "w") as f:
        pickle.dump([time_start, time_stop, num_aps_frame, num_dvs_events],
                    f)
        f.close()
    print ("[MESSAGE] Total duration of time:", (time_stop-time_start)/1e6)
    print ("[MESSAGE] Total number of frames:", num_aps_frame)
    print ("[MESSAGE] Total number of events:", num_dvs_events)
    print ("[MESSAGE] Frame rate:", num_aps_frame/((time_stop-time_start)/1e6))
    print ("[MESSAGE] Event rate:", num_dvs_events/((time_stop-time_start)/1e6))

    print('[DEBUG] sys_ts/time_stop', sys_ts, time_stop*1e-6)
    merged.exit.set()
    print('[DEBUG] output done')
    while not merged.done.is_set():
        print('[DEBUG] waiting for merger')
        time.sleep(1)
    print('[DEBUG] merger done')
    f_in.join()
    print('[DEBUG] stream joined')
    merged.join()
    print('[DEBUG] merger joined')

    del f_in
    del merged
    time.sleep(1)

os._exit(0)
