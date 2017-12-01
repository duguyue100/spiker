"""Support for ROS Bag.

WARNING: support with ROS installation.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from spiker.log import logger

try:
    import rosbag
except ImportError:
    logger.warning("You need to install ROS to enable this module.")


def get_topics(bag):
    """Get the list of topics from a ROS bag.

    Parameters
    ----------
    """
    pass
