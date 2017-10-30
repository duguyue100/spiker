"""Utility Function at Spiker Package Level.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function

import os

from spiker import log


def makedir(path, log_msg=None, verbose=False):
    """Make directory.

    Parameters
    ----------
    path : string
        the path of the directory.
        absolute path perferred.
    log_msg : str
        print log message if there is any.
    """
    try:
        os.makedirs(path)
        log.log(log_msg)
    except OSError:
        if verbose is True:
            log.log("The path %s exists" % path)


def makefile(file_path, log_msg=None, verbose=False):
    """Make file.

    Parameters
    ----------
    file_path : string
        the path of the file.
        absolute path perferred
    log_msg : string
        print log message if there is any
    """
    if not os.path.isfile(file_path):
        with open(file_path, "a"):
            os.utime(file_path, None)
        log.log(log_msg)
    else:
        if verbose is True:
            log.log("The file %s exists" % file_path)


def makedirs(path_list, log_msg=None, verbose=False):
    """Make a list of directories.

    Parameters
    ----------
    path_list : list
        list of directory paths.
        absolute path perferred.
    log_msg : string
        optional log mssage.
    """
    for path in path_list:
        makedir(path, verbose=verbose)
    log.log(log_msg)


def makefiles(filepath_list, log_msg=None, verbose=False):
    """Make a list of files.

    Parameters
    ----------
    filepath_list : list
        list of file paths.
        absolute path perferred.
    log_msg : string
        optional log mssage.
    """
    for filename in filepath_list:
        makefile(filename, verbose=verbose)
    log.log(log_msg)
