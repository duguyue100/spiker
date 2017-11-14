"""Export DDD17+ data fields.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle

import numpy as np

import spiker
from spiker.data import ddd17

# import csv file
csv_file_path = os.path.join(spiker.SPIKER_DATA, "ddd17", "DDD17plus.csv")
data_path_base = "/run/user/1000/gvfs/smb-share:server=" \
    "sensors-nas.lan.ini.uzh.ch,share=resiliosync/" \
    "DDD17-fordfocus/fordfocus"
# read csv file
recording_list = np.loadtxt(csv_file_path, dtype=str, delimiter=",")[:, 1]

# import list of fields
data_fields = list(ddd17.EXPORT_DATA_VI)

for data_item in recording_list:
    record_path = os.path.join(data_path_base, data_item)
    data_fields_dict = ddd17.export_data_field(record_path, data_fields)

    # save data at local
    save_path = os.path.join(spiker.SPIKER_EXTRA, "exported-data",
                             data_item[6:-5]+".pkl")
    with open(save_path, "w") as f:
        pickle.dump(data_fields_dict, f)
        f.close()
    print ("[MESSAGE] DATA %s Exported and Saved." % (data_item))
