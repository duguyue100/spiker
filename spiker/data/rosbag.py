"""Support for ROS Bag.

Support wrappers and easy-to-use functions to access a ROS bag.

WARNING: support with ROS installation.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os
import sys
import yaml

import spiker
from spiker import log
if os.name == 'posix' and sys.version_info[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess

logger = log.get_logger("rosbag-module", spiker.LOG_LEVEL)

try:
    import rosbag
    from cv_bridge import CvBridge, CvBridgeError
    frame_bridge = CvBridge()
except ImportError:
    logger.warning("You need to install ROS to enable this module.")


def get_topics(bag, topic_filters=None):
    """Get the list of topics from a ROS bag.

    # Parameters
        bag: rosbag.Bag
            a target rosbag.
        topics_filter: list
            list of topics that is/are filtered.
    # Returns
        bag_topics : dictionary
            dictionary of topics in the bag.
    """
    return bag.get_type_and_topic_info(topic_filters).topics


def get_yaml_description(bag_path):
    """Get a the bag discription in yaml format.

    # Parameters
        bag: str
            the absolute path of the bag file.
    # Returns
        info_dict: dictionary
            dictionary that contains the YAML description.
    """
    return yaml.load(subprocess.Popen(
        ["rosbag", "info", "--yaml", bag_path],
        std=subprocess.PIPE).communicate()[0])


def get_image(msg, encoding="mono8"):
    """Get image from a valid message.

    # Parameters
    msg : a ROS message that contains image data

    # Returns
    frame : numpy.ndarray
        numpy array
    """
    return frame_bridge.imgmsg_to_cv2(msg, encoding)


def get_msg_count(bag, topic):
    """Get number of instances for the topic.

    # Parameters
    bag : rosbag.Bag
        a target rosbag
    topic : string
        a valid topic name

    # Returns
    num_msgs : int
        number of messages in the topic
    """
    return bag.get_message_count(topic)
