"""Testing basic utilities of rosbag.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import rosbag

import spiker
from spiker import log
from spiker.data import rosbag as rb

logger = log.get_logger("rosbag-test", log.INFO)

bag_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "chalk_1.bag")

bag = rosbag.Bag(bag_path)

logger.info(rb.get_topics(bag))

#  for topic, msg, t in bag.read_messages():
#      print (topic)