"""Create Figures and Extract Results for CVPR paper.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
from os.path import join, isdir, isfile
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


#  option = "get-full-results"
#  option = "get-dvs-results"
#  option = "get-aps-results"
option = "get-loss-curves"

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
elif option == "get-loss-curves":
    sensor_modes = ["full", "dvs", "aps"]
    exps_dict = OrderedDict()

    for sensor_mode in sensor_modes:
        # read logs
        steer_night_logs = get_log_file_dict("night", sensor_mode, "steering")
        accel_night_logs = get_log_file_dict("night", sensor_mode, "accel")
        brake_night_logs = get_log_file_dict("night", sensor_mode, "brake")
        steer_day_logs = get_log_file_dict("day", sensor_mode, "steering")
        accel_day_logs = get_log_file_dict("day", sensor_mode, "accel")
        brake_day_logs = get_log_file_dict("day", sensor_mode, "brake")

        exps_logs = OrderedDict(
            steer_night_logs.items()+steer_day_logs.items() +
            accel_night_logs.items()+accel_day_logs.items() +
            brake_night_logs.items()+brake_day_logs.items())

        for env in ["night", "day"]:
            env_range = 7 if env == "night" else 8
            for env_idx in xrange(env_range):
                for task in ["steering", "accel", "brake"]:
                    log_name = task+"-"+env+"-%d-" % (env_idx+1)+sensor_mode
                    if log_name in exps_logs:
                        # load history
                        log_file = join(spiker.SPIKER_EXPS, log_name,
                                        "csv_history.log")
                        if isfile(log_file):
                            csv_log = utils.parse_csv_log(log_file)
                            # place the csv log to the right node

                            exps_dict[
                                env+"-%d" % (env_idx)][
                                    task][sensor_mode] = csv_log

        # collect curves
        for env in ["night", "day"]:
            env_range = 7 if env == "night" else 8
            for env_idx in xrange(env_range):
                # create grid specs
                fig = plt.figure(figsize=(12, 8))
                outer_grid = gridspec.GridSpec(3, 1, hspace=0.2)
                grid_idx = {"steering": 0, "accel": 1, "brake": 2}
                grid_title = {
                    "steering": "Prediction of Steering Wheel Angles",
                    "accel": "Prediction of Accelerator Pedal Position",
                    "brake": "Prediction of Brake Pedal Status"}

                for task in ["steering", "accel", "brake"]:
                    inner_grid = gridspec.GridSpecFromSubplotSpec(
                        2, 2, subplot_spec=grid_idx[task],
                        wspace=0.1, hspace=0.1)
                    # read log
                    full_log_dict = exps_dict[
                        env+"-%d" % (env_idx)][
                            task]["full"]
                    dvs_log_dict = exps_dict[
                        env+"-%d" % (env_idx)][
                            task]["dvs"]
                    aps_log_dict = exps_dict[
                        env+"-%d" % (env_idx)][
                            task]["aps"]
                    # training loss for the experiment
                    full_train_loss = full_log_dict["loss"]
                    dvs_train_loss = dvs_log_dict["loss"]
                    aps_train_loss = aps_log_dict["loss"]
                    # testing loss for the experiment
                    full_test_loss = full_log_dict["val_loss"]
                    dvs_test_loss = dvs_log_dict["val_loss"]
                    aps_test_loss = aps_log_dict["val_loss"]

                    if task == "brake":
                        # training acc for the experiment
                        full_train_mse = full_log_dict["binary_accuracy"]*100.
                        dvs_train_mse = dvs_log_dict["binary_accuracy"]*100.
                        aps_train_mse = aps_log_dict["binary_accuracy"]*100.
                        # testing mse for the experiment
                        full_test_mse = \
                            full_log_dict["val_binary_accuracy"]*100.
                        dvs_test_mse = dvs_log_dict["val_binary_accuracy"]*100.
                        aps_test_mse = aps_log_dict["val_binary_accuracy"]*100.
                    else:
                        # training mse for the experiment
                        full_train_mse = full_log_dict["mean_squared_error"]
                        dvs_train_mse = dvs_log_dict["mean_squared_error"]
                        aps_train_mse = aps_log_dict["mean_squared_error"]
                        # testing mse for the experiment
                        full_test_mse = full_log_dict["val_mean_squared_error"]
                        dvs_test_mse = dvs_log_dict["val_mean_squared_error"]
                        aps_test_mse = aps_log_dict["val_mean_squared_error"]

                    full_axis = range(1, full_train_loss.shape[0]+1)
                    dvs_axis = range(1, dvs_train_loss.shape[0]+1)
                    aps_axis = range(1, aps_train_loss.shape[0]+1)

                    # plot train loss figure
                    ax_trloss = plt.subplot(fig, inner_grid[0, 0])
                    ax_trloss.plot(
                        full_axis, full_train_loss, label="DVS+APS",
                        color="#08306b", linestyle="-", linewidth=2)
                    ax_trloss.plot(
                        dvs_axis, dvs_train_loss, label="DVS",
                        color="#7f2704", linestyle="-", linewidth=2)
                    ax_trloss.plot(
                        aps_axis, aps_train_loss, label="APS",
                        color="#3f007d", linestyle="-", linewidth=2)
                    plt.xlabel("epochs", fontsize=15)
                    plt.ylabel("loss", fontsize=15)
                    plt.title("Training Loss")
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.grid(linestyle="-.")
                    plt.legend(fontsize=15, shadow=True)

                    # plot test loss figure
                    ax_teloss = plt.subplot(fig, inner_grid[0, 1])
                    ax_teloss.plot(
                        full_axis, full_test_loss, label="DVS+APS",
                        color="#08306b", linestyle="-", linewidth=2)
                    ax_teloss.plot(
                        dvs_axis, dvs_test_loss, label="DVS",
                        color="#7f2704", linestyle="-", linewidth=2)
                    ax_teloss.plot(
                        aps_axis, aps_test_loss, label="APS",
                        color="#3f007d", linestyle="-", linewidth=2)
                    plt.xlabel("epochs", fontsize=15)
                    plt.ylabel("loss", fontsize=15)
                    plt.title("Testing Loss")
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.grid(linestyle="-.")
                    plt.legend(fontsize=15, shadow=True)

                    # plot train mse figure
                    ax_trmse = plt.subplot(fig, inner_grid[1, 0])
                    ax_trmse.plot(
                        full_axis, full_train_mse, label="DVS+APS",
                        color="#08306b", linestyle="--", linewidth=2)
                    ax_trmse.plot(
                        dvs_axis, dvs_train_mse, label="DVS",
                        color="#7f2704", linestyle="--", linewidth=2)
                    ax_trmse.plot(
                        aps_axis, aps_train_mse, label="APS",
                        color="#3f007d", linestyle="--", linewidth=2)
                    plt.xlabel("epochs", fontsize=15)
                    if task == "brake":
                        plt.ylabel("accuracy (%)", fontsize=15)
                        plt.title("Training Accuracy")
                    else:
                        plt.ylabel("mse", fontsize=15)
                        plt.title("Training MSE")
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.grid(linestyle="-.")
                    plt.legend(fontsize=15, shadow=True)

                    # plot test mse figure
                    ax_temse = plt.subplot(fig, inner_grid[1, 1])
                    ax_temse.plot(
                        full_axis, full_test_mse, label="DVS+APS",
                        color="#08306b", linestyle="--", linewidth=2)
                    ax_temse.plot(
                        dvs_axis, dvs_test_mse, label="DVS",
                        color="#7f2704", linestyle="--", linewidth=2)
                    ax_temse.plot(
                        aps_axis, aps_test_mse, label="APS",
                        color="#3f007d", linestyle="--", linewidth=2)
                    plt.xlabel("epochs", fontsize=15)
                    if task == "brake":
                        plt.ylabel("accuracy (%)", fontsize=15)
                        plt.title("Testing Accuracy")
                    else:
                        plt.ylabel("mse", fontsize=15)
                        plt.title("Testing MSE")
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.grid(linestyle="-.")
                    plt.legend(fontsize=15, shadow=True)

                    fig.suptitle(grid_title[task], size=20)

                    # save figure
                    plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                                     log_name+".pdf"),
                                dpi=600, format="pdf",
                                bbox="tight", pad_inches=0.5)
