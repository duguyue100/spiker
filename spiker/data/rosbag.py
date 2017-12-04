"""Support for ROS Bag.

Support wrappers and easy-to-use functions to access a ROS bag.

WARNING: support with ROS installation.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import spiker
from spiker import log

logger = log.get_logger("rosbag-module", spiker.LOG_LEVEL)

try:
    import rosbag
except ImportError:
    logger.warning("You need to install ROS to enable this module.")


def get_topics(bag, topic_filters=None):
    """Get the list of topics from a ROS bag.

    # Parameters
        bag: rosbag.Bag
            a target rosbag
        topics_filter: list
            list of topics that is/are filtered
    # Returns
        bag_topics : dictionary
            dictionary of topics in the bag
    """
    return bag.get_type_and_topic_info(topic_filters).topics
