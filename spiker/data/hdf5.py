"""HDF5 related files.

This file contains a set of functions that related to read and write
HDF5 files.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import

import h5py

from spiker import log

logger = log.get_logger("data-hdf5", log.DEBUG)


def init_hdf5(file_path, mode="w", cam_type="davis"):
    """Init HDF5 file object.

    # Parameters
    file_path : str
        absolute path for the HDF5 file.
    mode : str
        w : for writing
        r : for reading
    cam_type : str
        davis : for DAVIS camera
        dvs   : for DVS camera

    # Returns
    dataset : h5py.File
        The file object of the given dataset
    """
    if mode == "w":
        dataset = h5py.File(file_path, mode=mode)
        dataset.create_group("dvs")
        dataset.create_group("extra")
        if cam_type == "davis":
            dataset.create_group("aps")
            dataset.create_group("imu")
    elif mode == "r":
        dataset = h5py.File(file_path, mode=mode)

    return dataset
