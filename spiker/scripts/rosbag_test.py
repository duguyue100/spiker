"""Testing basic utilities of rosbag.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import rosbag

import spiker
from spiker.log import logger

bag_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "chalk_1.bag")

bag = rosbag.Bag(bag_path)


#  for topic, msg, t in bag.read_messages():
#      print (topic)
