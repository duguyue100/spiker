"""Create Figures and Extract Results for CVPR paper.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
from os.path import join, isdir, isfile
from collections import OrderedDict

import numpy as np

import spiker
from spiker.models import utils


def get_best_result(log_dict, mode="regress"):
    """Get result from a list of log files."""
    logs = OrderedDict()
    sum_score = 0
    for log_item in log_dict:
        csv_log = utils.parse_csv_log(log_dict[log_item])
        if mode == "regress":
            logs[log_item] = np.min(csv_log["val_mean_squared_error"])
            sum_score += logs[log_item]
        elif mode == "class":
            logs[log_item] = np.max(csv_log["val_accuracy"])
            sum_score += logs[log_item]
        elif mode == "binary":
            logs[log_item] = np.max(csv_log["val_binary_accuracy"])
            sum_score += logs[log_item]

    return logs, sum_score


def get_log_file_dict(env="day", mode="full", task="steering"):
    """Get data."""
    data_range = 8 if env == "day" else 7
    log_file_dict = OrderedDict()
    for idx in xrange(data_range):
        file_base = task+"-"+env+"-%d-" % (idx+1)+mode
        log_file_dict[file_base] = join(spiker.SPIKER_EXPS, file_base,
                                        "csv_history.log")
    return log_file_dict


option = "get-full-results"
option = "get-dvs-results"
option = "get-aps-results"

if option == "get-full-results":
    steer_day_logs = get_log_file_dict("day", "full", "steering")
    accel_day_logs = get_log_file_dict("day", "full", "accel")
    brake_day_logs = get_log_file_dict("day", "full", "brake")
    steer_night_logs = get_log_file_dict("night", "full", "steering")
    accel_night_logs = get_log_file_dict("night", "full", "accel")
    brake_night_logs = get_log_file_dict("night", "full", "brake")

    # get results
    steer_day_res, steer_day_sum = get_best_result(steer_day_logs)
    accel_day_res, accel_day_sum = get_best_result(accel_day_logs)
    brake_day_res, brake_day_sum = get_best_result(
        brake_day_logs, mode="binary")
    steer_night_res, steer_night_sum = get_best_result(steer_night_logs)
    accel_night_res, accel_night_sum = get_best_result(accel_night_logs)
    brake_night_res, brake_night_sum = get_best_result(
        brake_night_logs, mode="binary")

    print ("-"*30)
    print (steer_night_res)
    print (steer_day_res)
    print ((steer_day_sum+steer_night_sum)/15)
    print ("-"*30)
    print (accel_night_res)
    print (accel_day_res)
    print ((accel_day_sum+accel_night_sum)/15)
    print ("-"*30)
    print (brake_night_res)
    print (brake_day_res)
    print ("-"*30)
    print ((brake_day_sum+brake_night_sum)/15)
elif option == "get-dvs-results":
    steer_day_logs = get_log_file_dict("day", "dvs", "steering")
    accel_day_logs = get_log_file_dict("day", "dvs", "accel")
    brake_day_logs = get_log_file_dict("day", "dvs", "brake")
    steer_night_logs = get_log_file_dict("night", "dvs", "steering")
    accel_night_logs = get_log_file_dict("night", "dvs", "accel")
    brake_night_logs = get_log_file_dict("night", "dvs", "brake")

    # get results
    steer_day_res, steer_day_sum = get_best_result(steer_day_logs)
    accel_day_res, accel_day_sum = get_best_result(accel_day_logs)
    brake_day_res, brake_day_sum = get_best_result(
        brake_day_logs, mode="binary")
    steer_night_res, steer_night_sum = get_best_result(steer_night_logs)
    accel_night_res, accel_night_sum = get_best_result(accel_night_logs)
    brake_night_res, brake_night_sum = get_best_result(
        brake_night_logs, mode="binary")

    print ("-"*30)
    print (steer_night_res)
    print (steer_day_res)
    print ((steer_day_sum+steer_night_sum)/15)
    print ("-"*30)
    print (accel_night_res)
    print (accel_day_res)
    print ((accel_day_sum+accel_night_sum)/15)
    print ("-"*30)
    print (brake_night_res)
    print (brake_day_res)
    print ("-"*30)
    print ((brake_day_sum+brake_night_sum)/15)
elif option == "get-aps-results":
    steer_day_logs = get_log_file_dict("day", "aps", "steering")
    accel_day_logs = get_log_file_dict("day", "aps", "accel")
    brake_day_logs = get_log_file_dict("day", "aps", "brake")
    steer_night_logs = get_log_file_dict("night", "aps", "steering")
    accel_night_logs = get_log_file_dict("night", "aps", "accel")
    brake_night_logs = get_log_file_dict("night", "aps", "brake")

    # get results
    steer_day_res, steer_day_sum = get_best_result(steer_day_logs)
    accel_day_res, accel_day_sum = get_best_result(accel_day_logs)
    brake_day_res, brake_day_sum = get_best_result(
        brake_day_logs, mode="binary")
    steer_night_res, steer_night_sum = get_best_result(steer_night_logs)
    accel_night_res, accel_night_sum = get_best_result(accel_night_logs)
    brake_night_res, brake_night_sum = get_best_result(
        brake_night_logs, mode="binary")

    print ("-"*30)
    print (steer_night_res)
    print (steer_day_res)
    print ((steer_day_sum+steer_night_sum)/15)
    print ("-"*30)
    print (accel_night_res)
    print (accel_day_res)
    print ((accel_day_sum+accel_night_sum)/15)
    print ("-"*30)
    print (brake_night_res)
    print (brake_day_res)
    print ("-"*30)
    print ((brake_day_sum+brake_night_sum)/15)
