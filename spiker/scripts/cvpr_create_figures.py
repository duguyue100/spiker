"""Create Figures and Extract Results for CVPR paper.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
from os.path import join, isdir, isfile
from collections import OrderedDict
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spiker
from spiker.models import utils
from spiker.data import ddd17


def compute_log_curve(log_name, num_runs=4):
    """Compute log curve, provide mean and standard deviation."""
    log_collector = []
    for run_idx in range(1, num_runs+1):
        # prepare log path
        log_file = join(spiker.SPIKER_EXPS+"-run-%d" % (run_idx),
                        log_name, "csv_history.log")
        log_dict = utils.parse_csv_log(log_file)
        log_collector.append(log_dict)

    # compute for train loss
    train_loss = np.vstack(
        (log_collector[0]["loss"][np.newaxis, ...],
         log_collector[1]["loss"][np.newaxis, ...],
         log_collector[2]["loss"][np.newaxis, ...],
         log_collector[3]["loss"][np.newaxis, ...]))
    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)

    # compute for test loss
    test_loss = np.vstack(
        (log_collector[0]["val_loss"][np.newaxis, ...],
         log_collector[1]["val_loss"][np.newaxis, ...],
         log_collector[2]["val_loss"][np.newaxis, ...],
         log_collector[3]["val_loss"][np.newaxis, ...]))
    test_loss_mean = np.mean(test_loss, axis=0)
    test_loss_std = np.std(test_loss, axis=0)

    # compute for train mse
    train_mse = np.vstack(
        (log_collector[0]["mean_squared_error"][np.newaxis, ...],
         log_collector[1]["mean_squared_error"][np.newaxis, ...],
         log_collector[2]["mean_squared_error"][np.newaxis, ...],
         log_collector[3]["mean_squared_error"][np.newaxis, ...]))
    train_mse_mean = np.mean(train_mse, axis=0)
    train_mse_std = np.std(train_mse, axis=0)

    # compute for test mse
    test_mse = np.vstack(
        (log_collector[0]["val_mean_squared_error"][np.newaxis, ...],
         log_collector[1]["val_mean_squared_error"][np.newaxis, ...],
         log_collector[2]["val_mean_squared_error"][np.newaxis, ...],
         log_collector[3]["val_mean_squared_error"][np.newaxis, ...]))
    test_mse_mean = np.mean(test_mse, axis=0)
    test_mse_std = np.std(test_mse, axis=0)

    trloss = (train_loss_mean, train_loss_std)
    teloss = (test_loss_mean, test_loss_std)
    trmse = (train_mse_mean, train_mse_std)
    temse = (test_mse_mean, test_mse_std)

    return trloss, teloss, trmse, temse


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


def get_log_file_dict(env="day", mode="full", task="steering",
                      exp_dir=spiker.SPIKER_EXPS):
    """Get data."""
    data_range = 8 if env == "day" else 7
    log_file_dict = OrderedDict()
    for idx in xrange(data_range):
        file_base = task+"-"+env+"-%d-" % (idx+1)+mode
        log_file_dict[file_base] = join(exp_dir, file_base,
                                        "csv_history.log")
    return log_file_dict


#  option = "get-full-results"
#  option = "get-dvs-results"
#  option = "get-aps-results"
#  option = "get-loss-curves"
#  option = "get-results-reproduce"
#  option = "get-results-reproduce-steer"
#  option = "get-steering-results"
#  option = "attribute-hist"
option = "get-steer-loss-curves"

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
    for key in steer_night_res:
        print (key, ":", np.sqrt(steer_night_res[key])*180/np.pi)
    for key in steer_day_res:
        print (key, ":", np.sqrt(steer_day_res[key])*180/np.pi)
    print (np.sqrt((steer_day_sum+steer_night_sum)/15)*180/np.pi)
    print ("-"*30)
    for key in accel_night_res:
        print (key, ":", np.sqrt(accel_night_res[key])*100)
    for key in accel_day_res:
        print (key, ":", np.sqrt(accel_day_res[key])*100)
    print (np.sqrt((accel_day_sum+accel_night_sum)/15)*100)
    print ("-"*30)
    print (brake_night_res)
    print (brake_day_res)
    print ("-"*30)
    print ((brake_day_sum+brake_night_sum)/15)
elif option == "get-steering-results":
    sensor_mode = "full"
    # collecting logs
    # run 1
    day_logs_1 = get_log_file_dict("day", sensor_mode, "steering",
                                   spiker.SPIKER_EXPS+"-run-1")
    night_logs_1 = get_log_file_dict("night", sensor_mode, "steering",
                                     spiker.SPIKER_EXPS+"-run-1")
    # run 2 
    day_logs_2 = get_log_file_dict("day", sensor_mode, "steering",
                                   spiker.SPIKER_EXPS+"-run-2")
    night_logs_2 = get_log_file_dict("night", sensor_mode, "steering",
                                     spiker.SPIKER_EXPS+"-run-2")
    # run 3 
    day_logs_3 = get_log_file_dict("day", sensor_mode, "steering",
                                   spiker.SPIKER_EXPS+"-run-3")
    night_logs_3 = get_log_file_dict("night", sensor_mode, "steering",
                                     spiker.SPIKER_EXPS+"-run-3")
    # run 3 
    day_logs_4 = get_log_file_dict("day", sensor_mode, "steering",
                                   spiker.SPIKER_EXPS+"-run-4")
    night_logs_4 = get_log_file_dict("night", sensor_mode, "steering",
                                     spiker.SPIKER_EXPS+"-run-4")
    # collect results
    day_res_1, day_sum_1 = get_best_result(day_logs_1)
    night_res_1, night_sum_1 = get_best_result(night_logs_1)
    day_res_2, day_sum_2 = get_best_result(day_logs_2)
    night_res_2, night_sum_2 = get_best_result(night_logs_2)
    day_res_3, day_sum_3 = get_best_result(day_logs_3)
    night_res_3, night_sum_3 = get_best_result(night_logs_3)
    day_res_4, day_sum_4 = get_best_result(day_logs_4)
    night_res_4, night_sum_4 = get_best_result(night_logs_4)

    # calculate mean and variance
    for key in night_res_1:
        temp_res = np.array([night_res_1[key], night_res_2[key],
                             night_res_3[key],
                             night_res_4[key]])
        temp_res = np.sqrt(temp_res)*180/np.pi
        print (key, ":", temp_res.mean(), temp_res.std())
    for key in day_res_1:
        temp_res = np.array([day_res_1[key], day_res_2[key], day_res_3[key],
                             day_res_4[key]])
        temp_res = np.sqrt(temp_res)*180/np.pi
        print (key, ":", temp_res.mean(), "std:", temp_res.std(),
               "best", temp_res.argmin())
        print (temp_res)
    avg_error = np.array([day_sum_1+night_sum_1,
                          day_sum_2+night_sum_2,
                          day_sum_3+night_sum_3,
                          day_sum_4+night_sum_4])/15.
    avg_error = np.sqrt(avg_error)*180/np.pi
    print ("Average Error:", avg_error.mean(), "std:", avg_error.std())
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
    for key in steer_night_res:
        print (key, ":", np.sqrt(steer_night_res[key])*180/np.pi)
    for key in steer_day_res:
        print (key, ":", np.sqrt(steer_day_res[key])*180/np.pi)
    print (np.sqrt((steer_day_sum+steer_night_sum)/15)*180/np.pi)
    print ("-"*30)
    for key in accel_night_res:
        print (key, ":", np.sqrt(accel_night_res[key])*100)
    for key in accel_day_res:
        print (key, ":", np.sqrt(accel_day_res[key])*100)
    print (np.sqrt((accel_day_sum+accel_night_sum)/15)*100)
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
    for key in steer_night_res:
        print (key, ":", np.sqrt(steer_night_res[key])*180/np.pi)
    for key in steer_day_res:
        print (key, ":", np.sqrt(steer_day_res[key])*180/np.pi)
    print (np.sqrt((steer_day_sum+steer_night_sum)/15)*180/np.pi)
    print ("-"*30)
    for key in accel_night_res:
        print (key, ":", np.sqrt(accel_night_res[key])*100)
    for key in accel_day_res:
        print (key, ":", np.sqrt(accel_day_res[key])*100)
    print (np.sqrt((accel_day_sum+accel_night_sum)/15)*100)
    print ("-"*30)
    print (brake_night_res)
    print (brake_day_res)
    print ("-"*30)
    print ((brake_day_sum+brake_night_sum)/15)
elif option == "get-loss-curves":
    # collect curves
    for env in ["night", "day"]:
        env_range = 7 if env == "night" else 8
        for env_idx in xrange(env_range):
            # create grid specs
            fig = plt.figure(figsize=(16, 24))
            outer_grid = gridspec.GridSpec(3, 1, hspace=0.4)
            grid_idx = {"steering": 0, "accel": 1, "brake": 2}
            log_name = env+"-%d" % (env_idx+1)

            for task in ["steering", "accel", "brake"]:
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=outer_grid[grid_idx[task]],
                    wspace=0.3, hspace=0.8)
                # read log
                full_log_name = task+"-"+env+"-%d-" % (env_idx+1)+"full"
                dvs_log_name = task+"-"+env+"-%d-" % (env_idx+1)+"dvs"
                aps_log_name = task+"-"+env+"-%d-" % (env_idx+1)+"aps"
                full_log_file = join(spiker.SPIKER_EXPS, full_log_name,
                                     "csv_history.log")
                dvs_log_file = join(spiker.SPIKER_EXPS, dvs_log_name,
                                    "csv_history.log")
                aps_log_file = join(spiker.SPIKER_EXPS, aps_log_name,
                                    "csv_history.log")
                full_log_dict = utils.parse_csv_log(full_log_file)
                dvs_log_dict = utils.parse_csv_log(dvs_log_file)
                aps_log_dict = utils.parse_csv_log(aps_log_file)
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
                ax_trloss = plt.Subplot(fig, inner_grid[0, 0])
                fig.add_subplot(ax_trloss)
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
                plt.title("Training Loss ("+task+")")
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(linestyle="-.")
                #  plt.legend(fontsize=15, shadow=True)
                plt.legend(shadow=True, loc=1)

                # plot test loss figure
                ax_teloss = plt.Subplot(fig, inner_grid[0, 1])
                fig.add_subplot(ax_teloss)
                ax_teloss.plot(
                    full_axis, full_test_loss, label="DVS+APS",
                    color="#08306b", linestyle="--", linewidth=2)
                ax_teloss.plot(
                    dvs_axis, dvs_test_loss, label="DVS",
                    color="#7f2704", linestyle="--", linewidth=2)
                ax_teloss.plot(
                    aps_axis, aps_test_loss, label="APS",
                    color="#3f007d", linestyle="--", linewidth=2)
                ax_teloss.set_yscale("log")
                plt.xlabel("epochs", fontsize=15)
                plt.ylabel("loss", fontsize=15)
                plt.title("Testing Loss ("+task+")")
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(linestyle="-.")
                #  plt.legend(fontsize=15, shadow=True)
                plt.legend(shadow=True, loc=1)

                # plot train mse figure
                ax_trmse = plt.Subplot(fig, inner_grid[1, 0])
                fig.add_subplot(ax_trmse)
                ax_trmse.plot(
                    full_axis, full_train_mse, label="DVS+APS",
                    color="#08306b", linestyle="-", linewidth=2)
                ax_trmse.plot(
                    dvs_axis, dvs_train_mse, label="DVS",
                    color="#7f2704", linestyle="-", linewidth=2)
                ax_trmse.plot(
                    aps_axis, aps_train_mse, label="APS",
                    color="#3f007d", linestyle="-", linewidth=2)
                plt.xlabel("epochs", fontsize=15)
                if task == "brake":
                    plt.ylabel("accuracy (%)", fontsize=15)
                    plt.title("Training Accuracy ("+task+")")
                else:
                    plt.ylabel("mse", fontsize=15)
                    plt.title("Training MSE ("+task+")")
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(linestyle="-.")
                #  plt.legend(fontsize=15, shadow=True)
                plt.legend(shadow=True, loc=1)

                # plot test mse figure
                ax_temse = plt.Subplot(fig, inner_grid[1, 1])
                fig.add_subplot(ax_temse)
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
                    plt.title("Testing Accuracy ("+task+")")
                else:
                    ax_temse.set_yscale("log")
                    plt.ylabel("mse", fontsize=15)
                    plt.title("Testing MSE ("+task+")")
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.grid(linestyle="-.")
                #  plt.legend(fontsize=15, shadow=True)
                plt.legend(shadow=True, loc=1)

            # save figure
            plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                             log_name+".pdf"),
                        dpi=600, format="pdf",
                        bbox="tight", pad_inches=0.5)
elif option == "get-results-reproduce":
    data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                             "jul29/rec1501349894-export.hdf5")
    frame_cut = [200, 1500]
    # load model
    model_base = "-day-5-"
    exp_type = ["steering", "accel", "brake"]
    sensor_type = ["full", "dvs", "aps"]

    # get prediction
    load_prediction = os.path.join(
        spiker.SPIKER_EXTRA, "pred"+model_base+"result")

    if os.path.isfile(load_prediction):
        print ("[MESSAGE] Prediction available")
        with open(load_prediction, "r") as f:
            (steer_full, steer_dvs, steer_aps,
             accel_full, accel_dvs, accel_aps,
             brake_full, brake_dvs, brake_aps) = pickle.load(f)
            f.close()

    num_samples = 500
    frames, steering = ddd17.prepare_train_data(data_path,
                                                target_size=None,
                                                y_name="steering",
                                                frame_cut=frame_cut,
                                                data_portion="test",
                                                data_type="uint8",
                                                num_samples=num_samples)
    steering = ddd17.prepare_train_data(data_path,
                                        target_size=None,
                                        y_name="steering",
                                        only_y=True,
                                        frame_cut=frame_cut,
                                        data_portion="test",
                                        data_type="uint8")
    accel = ddd17.prepare_train_data(data_path,
                                     y_name="accel",
                                     only_y=True,
                                     frame_cut=frame_cut,
                                     data_portion="test")
    brake = ddd17.prepare_train_data(data_path,
                                     y_name="brake",
                                     only_y=True,
                                     frame_cut=frame_cut,
                                     data_portion="test")
    x_axis = np.arange(steering.shape[0])
    idx = 200

    fig = plt.figure(figsize=(10, 8))
    outer_grid = gridspec.GridSpec(2, 1, wspace=0.1)

    # plot frames
    frame_grid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_grid[0, 0],
        hspace=0.1)
    aps_frame = plt.Subplot(fig, frame_grid[0])
    aps_frame.imshow(frames[idx, :, :, 1], cmap="gray")
    aps_frame.axis("off")
    aps_frame.set_title("APS Frame")
    fig.add_subplot(aps_frame)
    dvs_frame = plt.Subplot(fig, frame_grid[1])
    dvs_frame.imshow(frames[idx, :, :, 0], cmap="gray")
    dvs_frame.axis("off")
    dvs_frame.set_title("DVS Frame")
    fig.add_subplot(dvs_frame)

    # plot steering curve
    curve_grid = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_grid[1, 0],
        hspace=0.35)
    steering_curve = plt.Subplot(fig, curve_grid[0, 0])
    min_steer = np.min(steering*180/np.pi)
    max_steer = np.max(steering*180/np.pi)
    steering_curve.plot(x_axis, steering*180/np.pi,
                        label="groundtruth",
                        color="#08306b",
                        linestyle="-",
                        linewidth=2)
    steering_curve.plot(x_axis, steer_full*180/np.pi,
                        label="DVS+APS",
                        color="#7f2704",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(x_axis, steer_dvs*180/np.pi,
                        label="DVS",
                        color="#3f007d",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(x_axis, steer_aps*180/np.pi,
                        label="APS",
                        color="#00441b",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot((idx, idx), (min_steer, max_steer), color="black",
                        linestyle="-", linewidth=1)
    steering_curve.set_xlim(left=1, right=steering.shape[0])
    steering_curve.set_xticks([])
    steering_curve.set_title("Steering Wheel Angle Prediction")
    steering_curve.grid(linestyle="-.")
    steering_curve.legend(fontsize=8, loc=1)
    steering_curve.set_ylabel("degree")
    fig.add_subplot(steering_curve)

    # plot accel curve
    accel_curve = plt.Subplot(fig, curve_grid[1, 0])
    min_accel = np.min(accel*100)
    max_accel = np.max(accel*100)
    accel_curve.plot(x_axis, accel*100,
                     label="groundtruth",
                     color="#08306b",
                     linestyle="-",
                     linewidth=2)
    accel_curve.plot(x_axis, accel_full*100,
                     label="DVS+APS",
                     color="#7f2704",
                     linestyle="-",
                     linewidth=1)
    accel_curve.plot(x_axis, accel_dvs*100,
                     label="DVS",
                     color="#3f007d",
                     linestyle="-",
                     linewidth=1)
    accel_curve.plot(x_axis, accel_aps*100,
                     label="APS",
                     color="#00441b",
                     linestyle="-",
                     linewidth=1)
    accel_curve.plot((idx, idx), (min_accel, max_accel), color="black",
                     linestyle="-", linewidth=1)
    accel_curve.set_xlim(left=1, right=accel.shape[0])
    accel_curve.set_xticks([])
    accel_curve.set_title("Accelerator Pedal Position Prediction")
    accel_curve.grid(linestyle="-.")
    accel_curve.legend(fontsize=8, loc=1)
    accel_curve.set_ylabel("pressure (%)")
    fig.add_subplot(accel_curve)

    # plot brake curve
    brake_curve = plt.Subplot(fig, curve_grid[2, 0])
    brake_curve.plot(x_axis, brake*100,
                     label="groundtruth",
                     color="#08306b",
                     linestyle=" ",
                     marker="o",
                     markersize=4)
    brake_curve.plot(x_axis, brake_full,
                     label="DVS+APS",
                     color="#7f2704",
                     linestyle=" ",
                     marker="o",
                     markersize=2)
    brake_curve.plot(x_axis, brake_dvs,
                     label="DVS",
                     color="#3f007d",
                     linestyle=" ",
                     marker="o",
                     markersize=2)
    brake_curve.plot(x_axis, brake_aps,
                     label="APS",
                     color="#00441b",
                     linestyle=" ",
                     marker="o",
                     markersize=2)
    brake_curve.plot((idx, idx), (0, 100), color="black",
                     linestyle="-", linewidth=1)
    brake_curve.set_xlim(left=1, right=brake.shape[0])
    brake_curve.set_yticks([0, 100])
    brake_curve.set_yticklabels(["OFF", "ON"])
    brake_curve.set_title("Brake Pedal Position Prediction")
    brake_curve.grid(linestyle="-.")
    brake_curve.legend(fontsize=8, loc=1)
    brake_curve.set_xlabel("frame")
    brake_curve.set_ylabel("ON/OFF")
    fig.add_subplot(brake_curve)

    plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                     model_base+"result"+".pdf"),
                dpi=600, format="pdf",
                bbox="tight", pad_inches=0.5)
elif option == "get-results-reproduce-steer":
    data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                             "jul29/rec1501349894-export.hdf5")
    origin_data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                                    "jul29/rec1501349894.hdf5")
    frame_cut = [200, 1500]
    # load model
    model_base = "-day-5-"
    exp_type = ["steering", "accel", "brake"]
    sensor_type = ["full", "dvs", "aps"]

    # get prediction
    load_prediction = os.path.join(
        spiker.SPIKER_EXTRA, "pred"+model_base+"result")

    if os.path.isfile(load_prediction):
        print ("[MESSAGE] Prediction available")
        with open(load_prediction, "r") as f:
            (steer_full, steer_dvs, steer_aps,
             accel_full, accel_dvs, accel_aps,
             brake_full, brake_dvs, brake_aps) = pickle.load(f)
            f.close()

    num_samples = 500
    frames, steering = ddd17.prepare_train_data(data_path,
                                                target_size=None,
                                                y_name="steering",
                                                frame_cut=frame_cut,
                                                data_portion="test",
                                                data_type="uint8",
                                                num_samples=num_samples)
    steering = ddd17.prepare_train_data(data_path,
                                        target_size=None,
                                        y_name="steering",
                                        only_y=True,
                                        frame_cut=frame_cut,
                                        data_portion="test",
                                        data_type="uint8")
    steer, steer_time = ddd17.export_data_field(
        origin_data_path, ['steering_wheel_angle'], frame_cut=frame_cut,
        data_portion="test")
    steer_time -= steer_time[0]
    # in ms
    steer_time = steer_time.astype("float32")/1e6
    print (steer_time)

    idx = 200

    fig = plt.figure(figsize=(10, 8))
    outer_grid = gridspec.GridSpec(2, 1, wspace=0.1)

    # plot frames
    frame_grid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_grid[0, 0],
        hspace=0.1)
    aps_frame = plt.Subplot(fig, frame_grid[0])
    aps_frame.imshow(frames[idx, :, :, 1], cmap="gray")
    aps_frame.axis("off")
    aps_frame.set_title("APS Frame")
    fig.add_subplot(aps_frame)
    dvs_frame = plt.Subplot(fig, frame_grid[1])
    dvs_frame.imshow(frames[idx, :, :, 0], cmap="gray")
    dvs_frame.axis("off")
    dvs_frame.set_title("DVS Frame")
    fig.add_subplot(dvs_frame)

    # plot steering curve
    steering_curve = plt.Subplot(fig, outer_grid[1, 0])
    min_steer = np.min(steering*180/np.pi)
    max_steer = np.max(steering*180/np.pi)
    steering_curve.plot(steer_time, steering*180/np.pi,
                        label="groundtruth",
                        color="#08306b",
                        linestyle="-",
                        linewidth=2)
    steering_curve.plot(steer_time, steer_full*180/np.pi,
                        label="DVS+APS",
                        color="#7f2704",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(steer_time, steer_dvs*180/np.pi,
                        label="DVS",
                        color="#3f007d",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot(steer_time, steer_aps*180/np.pi,
                        label="APS",
                        color="#00441b",
                        linestyle="-",
                        linewidth=1)
    steering_curve.plot((steer_time[idx], steer_time[idx]),
                        (min_steer, max_steer), color="black",
                        linestyle="-", linewidth=1)
    steering_curve.set_xlim(left=0, right=steer_time[-1])
    steering_curve.set_title("Steering Wheel Angle Prediction")
    steering_curve.grid(linestyle="-.")
    steering_curve.legend(fontsize=10)
    steering_curve.set_ylabel("degree")
    steering_curve.set_xlabel("time (s)")
    fig.add_subplot(steering_curve)

    plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                     "vis"+model_base+"result"+".pdf"),
                dpi=600, format="pdf",
                bbox="tight", pad_inches=0.5)
elif option == "attribute-hist":
    data_path = os.path.join(spiker.SPIKER_EXTRA, "exported-data")
    data_list = os.listdir(data_path)
    #  attribute = "steering_wheel_angle"
    #  attribute = "brake_pedal_status"
    #  attribute = "vehicle_speed"
    #  attribute = "headlamp_status"
    #  attribute = "accelerator_pedal_position"
    #  attribute = "engine_speed"
    #  attribute = "odometer"
    #  attribute = "torque_at_transmission"
    #  attribute = "transmission_gear_position"
    #  attribute = "fuel_level"
    #  attribute = "fuel_level"
    #  attribute = "high_beam_status"
    #  attribute = "windshield_wiper_status"
    #  attribute = "ignition_status"
    #  attribute = "parking_brake_status"
    #  attribute = "latitude"
    attribute = "longitude"

    # read data
    attribute_collector = []
    for item in data_list:
        file_path = os.path.join(data_path, item)
        with open(file_path, "r") as f:
            data_fields = pickle.load(f)
            f.close()
        data = data_fields[attribute]["data"].astype("float32")
        time_stamp = data_fields[attribute]["timestamp"].astype("float32")
        non_zero_idx = np.count_nonzero(time_stamp)
        data = data[:non_zero_idx]
        time_stamp = time_stamp[:non_zero_idx]
        attribute_collector.append(data)
        print ("[MESSAGE] Processed %s" % (item))

    # join all data
    all_data = attribute_collector[0]
    for idx in xrange(1, len(attribute_collector)):
        all_data = np.hstack((all_data, attribute_collector[idx]))

    # mse
    #  mse = np.sqrt(np.mean((all_data/180*np.pi)**2))*180/np.pi
    #  print ("MSE:", mse)

    weights = np.ones_like(all_data)/float(600)
    print (all_data.max())
    print (all_data.min())

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()
    ax.hist(all_data, bins=40, color="#666970", weights=weights,
             linewidth=1.2, edgecolor="black")
    #  ax.hist(all_data, bins=2, color="#666970",
    #           linewidth=1.2, edgecolor="black")
    # latitude
    plt.xlabel("degree", fontsize=15)
    plt.ylabel("time (min)", fontsize=15)
    # fuel level
    #  plt.xlabel("fuel level (%)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # transmission_gear_position
    #  plt.xlabel("gear position", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # torque_at_transmission
    #  plt.xlabel("torque (N x m)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # steering pedal status
    #  plt.xlabel("rotation (degree)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # vehicle speed
    #  plt.xlabel("speed (km/h)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # engine speed
    #  plt.xlabel("speed (rpm)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # odometer
    #  plt.xlabel("odometer (km)", fontsize=15)
    #  plt.ylabel("time (min)", fontsize=15)
    # brake pedal status
    #  plt.xticks([0.0, 1.0], ["released", "pressed"],
    #             rotation="vertical", fontsize=15)
    # headlamp_status
    #  plt.xticks([0.0, 1.0], ["OFF", "ON"],
    #             rotation="vertical", fontsize=15)
    # high_beam_status
    #  plt.xticks([0.0, 1.0], ["OFF", "ON"],
    #             rotation="vertical", fontsize=15)
    # windshield_wiper_status
    #  plt.xticks([0.0, 1.0], ["OFF", "ON"],
    #             rotation="vertical", fontsize=15)
    # ignition_status
    #  plt.xticks([0.0, 3.0], ["OFF", "ON"],
    #             rotation="vertical", fontsize=15)
    # parking_brake_status
    #  plt.xticks([0.0, 1.0], ["OFF", "ON"],
    #             rotation="vertical", fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                     attribute+".pdf"),
                dpi=600, format="pdf",
                pad_inches=0.5)
elif option == "get-steer-loss-curves":
    # collect curves
    for env in ["night", "day"]:
        env_range = 7 if env == "night" else 8
        for env_idx in xrange(env_range):
            # create grid specs
            log_name = "steering"+"-"+env+"-%d-" % (env_idx+1)
            full_log_name = "steering"+"-"+env+"-%d-" % (env_idx+1)+"full"
            dvs_log_name = "steering"+"-"+env+"-%d-" % (env_idx+1)+"dvs"
            aps_log_name = "steering"+"-"+env+"-%d-" % (env_idx+1)+"aps"

            full_log = compute_log_curve(full_log_name)
            dvs_log = compute_log_curve(dvs_log_name)
            aps_log = compute_log_curve(aps_log_name)

            full_axis = range(1, full_log[0][0].shape[0]+1)
            dvs_axis = range(1, dvs_log[0][0].shape[0]+1)
            aps_axis = range(1, aps_log[0][0].shape[0]+1)

            # produce figure 
            fig = plt.figure(figsize=(16, 8))

            print (full_log[0][0].shape)

            # train loss
            trloss_ax = fig.add_subplot(221)
            trloss_ax.plot(full_axis, full_log[0][0], label="DVS+APS",
                           color="#08306b", linestyle="-", linewidth=2)
            trloss_ax.plot(dvs_axis, dvs_log[0][0], label="DVS",
                           color="#7f2704", linestyle="-", linewidth=2)
            trloss_ax.plot(aps_axis, aps_log[0][0], label="APS",
                           color="#3f007d", linestyle="-", linewidth=2)
            plt.xlabel("epochs", fontsize=15)
            plt.ylabel("loss", fontsize=15)
            plt.title("Training Loss")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid(linestyle="-.")
            plt.yscale("log")
            plt.legend(fontsize=15, shadow=True)

            # test loss
            teloss_ax = fig.add_subplot(222)
            teloss_ax.plot(full_axis, full_log[1][0], label="DVS+APS",
                           color="#08306b", linestyle="-", linewidth=2)
            teloss_ax.plot(dvs_axis, dvs_log[1][0], label="DVS",
                           color="#7f2704", linestyle="-", linewidth=2)
            teloss_ax.plot(aps_axis, aps_log[1][0], label="APS",
                           color="#3f007d", linestyle="-", linewidth=2)
            plt.xlabel("epochs", fontsize=15)
            plt.ylabel("loss", fontsize=15)
            plt.title("Testing Loss")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid(linestyle="-.")
            plt.yscale("log")
            plt.legend(fontsize=15, shadow=True)

            # train mse
            trmse_ax = fig.add_subplot(223)
            trmse_ax.plot(full_axis, full_log[2][0], label="DVS+APS",
                          color="#08306b", linestyle="-", linewidth=2)
            trmse_ax.plot(dvs_axis, dvs_log[2][0], label="DVS",
                           color="#7f2704", linestyle="-", linewidth=2)
            trmse_ax.plot(aps_axis, aps_log[2][0], label="APS",
                           color="#3f007d", linestyle="-", linewidth=2)
            plt.xlabel("epochs", fontsize=15)
            plt.ylabel("mse", fontsize=15)
            plt.title("Training MSE")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid(linestyle="-.")
            plt.yscale("log")
            plt.legend(fontsize=15, shadow=True)

            # test mse
            temse_ax = fig.add_subplot(224)
            temse_ax.plot(full_axis, full_log[3][0], label="DVS+APS",
                          color="#08306b", linestyle="-", linewidth=2)
            temse_ax.plot(dvs_axis, dvs_log[3][0], label="DVS",
                           color="#7f2704", linestyle="-", linewidth=2)
            temse_ax.plot(aps_axis, aps_log[3][0], label="APS",
                           color="#3f007d", linestyle="-", linewidth=2)
            plt.xlabel("epochs", fontsize=15)
            plt.ylabel("mse", fontsize=15)
            plt.title("Testing MSE")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.grid(linestyle="-.")
            plt.yscale("log")
            plt.legend(fontsize=15, shadow=True)

            plt.tight_layout()

            # save figure
            plt.savefig(join(spiker.SPIKER_EXTRA, "cvprfigs",
                             log_name+".pdf"),
                        dpi=600, format="pdf",
                        bbox="tight", pad_inches=0.5)
