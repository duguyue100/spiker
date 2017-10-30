"""DDD17 Data Utility.

Migrate DDD17 data reading utilities from
old utils to here.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function


import numpy as np

from spiker import data

# data fields
EXPORT_DATA_VI = {
        'steering_wheel_angle',
        'brake_pedal_status',
        'accelerator_pedal_position',
        'engine_speed',
        'vehicle_speed',
        'windshield_wiper_status',
        'headlamp_status',
        'transmission_gear_position',
        'torque_at_transmission',
        'fuel_level',
        'high_beam_status',
        'ignition_status',
        # 'lateral_acceleration',
        'latitude',
        'longitude',
        # 'longitudinal_acceleration',
        'odometer',
        'parking_brake_status',
        # 'fine_odometer_since_restart',
        'fuel_consumed_since_restart',
    }

EXPORT_DATA_DAVIS = {
        'dvs_frame',
        'aps_frame',
    }

EXPORT_DATA = EXPORT_DATA_VI.union(EXPORT_DATA_DAVIS)


def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8


def raster_evts(data, data_shape=data.DAVIS346_SHAPE):
    _histrange = [(0, v) for v in data_shape]
    pol_on = data[:, 3] == 1
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            data[pol_on, 2], data[pol_on, 1],
            bins=data_shape, range=_histrange)
    img_off, _, _ = np.histogram2d(
            data[pol_off, 2], data[pol_off, 1],
            bins=data_shape, range=_histrange)
    return (img_on - img_off).astype(np.int16)
