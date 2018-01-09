"""Testing basic utilities of rosbag.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from builtins import range
import os

import numpy as np
import rosbag
import h5py

import spiker
from spiker import log
from spiker.data import rosbag as rb

logger = log.get_logger("rosbag-test", log.INFO)

bag_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_test.bag")
hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "ccw_foyer_record_12_12_17_test.hdf5")

bag = rosbag.Bag(bag_path, "r")

bag_topics = rb.get_topics(bag)

for key, value in bag_topics.iteritems():
    logger.info("Topic: %s" % (key))

num_images = rb.get_msg_count(bag, "/dvs/image_raw")
logger.info("Number of images: %d" % (num_images))
num_event_pkgs = rb.get_msg_count(bag, "/dvs/events")
logger.info("Number of event packets: %d" % (num_event_pkgs))
num_imu_pkgs = rb.get_msg_count(bag, "/dvs/imu")
logger.info("Number of IMU packets: %d" % (num_imu_pkgs))
num_pwm_pkgs = rb.get_msg_count(bag, "/raw_pwm")
logger.info("Number of pwm packets: %d" % (num_pwm_pkgs))
start_time = bag.get_start_time()  # time in second
logger.info("Start time: %f" % (start_time))
end_time = bag.get_end_time()
logger.info("End time: %f" % (end_time))
logger.info("Duration: %f s" % (end_time-start_time))

# image shape
img_shape = (180, 240)

# Define HDF5
dataset = h5py.File(hdf5_path, "w")
aps_group = dataset.create_group("aps")
dvs_group = dataset.create_group("dvs")
imu_group = dataset.create_group("imu")
extra_group = dataset.create_group("extra")
pwm_group = extra_group.create_group("pwm")

# define dataset
# APS Frame data
aps_frame_ds = aps_group.create_dataset(
    name="aps_data",
    shape=(num_images,)+img_shape,
    dtype="uint8")
aps_time_ds = aps_group.create_dataset(
    name="aps_ts",
    shape=(num_images,),
    dtype="int64")

# PWM data
pwm_data_ds = pwm_group.create_dataset(
    name="pwm_data",
    shape=(num_pwm_pkgs, 3),
    dtype="float32")
pwm_time_ds = pwm_group.create_dataset(
    name="pwm_ts",
    shape=(num_pwm_pkgs,),
    dtype="int64")

# DVS data
dvs_data_ds = dvs_group.create_dataset(
    name="event_loc",
    maxshape=(None, 2),
    dtype="uint16")
dvs_time_ds = dvs_group.create_dataset(
    name="event_ts",
    maxshape=(None,),
    dtype="int64")
dvs_pol_ds = dvs_group.create_dataset(
    name="event_pol",
    maxshape=(None,),
    dtype="bool")

# topic list
topics_list = ["/dvs/events"]
#  topics_list = ["/dvs/image_raw", "/dvs/events", "/dvs/imu",
#                 "/raw_pwm"]

frame_idx = 0
pwm_idx = 0
for topic, msg, t in bag.read_messages(topics=topics_list):
    if topic in ["/dvs/image_raw"]:
        image = rb.get_image(msg)

        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs
        # time in microsec
        time_stamp = str(secs)+str(nsecs)[:6]
        time_stamp = int(time_stamp)
        logger.info("Time: %s" % (time_stamp))

        aps_frame_ds[frame_idx] = image
        aps_time_ds[frame_idx] = time_stamp
        frame_idx += 1
    elif topic in ["/raw_pwm"]:
        steering = msg.steering
        throttle = msg.throttle
        gear_shift = msg.gear_shift
        time_stamp = t.to_nsec()//1000

        pwm_data_ds[pwm_idx] = np.array([steering, throttle, gear_shift])
        pwm_time_ds[pwm_idx] = time_stamp
        pwm_idx += 1
    elif topic in ["/dvs/events"]:
        events = msg.events
        num_events = len(events)
        events_loc_arr = np.zeros((num_events, 2), dtype=np.uint16)
        events_ts_arr = np.zeros((num_events,), dtype=np.int64)
        events_pol_arr = np.zeros((num_events,), dtype=np.bool)

        for event_idx in range(num_events):
            # event location
            events_loc_arr[event_idx, 0] = events[event_idx].x
            events_loc_arr[event_idx, 1] = events[event_idx].y

            # event timestamp
            time_stamp = str(events[event_idx].secs) + \
                str(events[event_idx].nsecs)[:6]
            time_stamp = int(time_stamp)
            events_ts_arr[event_idx] = time_stamp

            # event polarity
            events_pol_arr[event_idx] = events[event_idx].polarity

        dvs_data_ds.resize(dvs_data_ds.shape[0]+num_events,
                           axis=0)
        dvs_time_ds.resize(dvs_time_ds.shape[0]+num_events,
                           axis=0)
        dvs_pol_ds.resize(dvs_pol_ds.shape[0]+num_events,
                          axis=0)

        dvs_data_ds[:-num_events] = events_loc_arr
        dvs_time_ds[:-num_events] = events_ts_arr
        dvs_pol_ds[:-num_events] = events_pol_arr

dataset.close()
