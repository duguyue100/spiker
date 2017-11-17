"""Export DDD17 Experiment Results.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
import cPickle as pickle

import spiker
from spiker.data import ddd17
from spiker.models import utils


def find_best(exp_dir, run_root):
    """find best experiment."""
    exp_dir = os.path.join(spiker.SPIKER_EXPS, exp_dir)
    file_list = os.listdir(exp_dir)
    file_clean_list = []
    for item in file_list:
        if ".hdf5" in item:
            file_clean_list.append(item)
    max_length = 0
    for item in file_clean_list:
        if len(item) > max_length:
            max_length = len(item)
    # filter the longest ones
    file_max_list = []
    for item in file_clean_list:
        if len(item) == max_length:
            file_max_list.append(item)

    file_list = sorted(file_max_list)
    return file_list[-1]


def get_prediction(X_test, exp_type, model_base, sensor_type, exp_dir):
    """Get prediction."""
    model_file_base = exp_type+model_base+sensor_type
    model_path = os.path.join(
        exp_dir, model_file_base,
        find_best(model_file_base, exp_dir))
    print ("[MESSAGE]", model_path)
    model = utils.keras_load_model(model_path)
    prediction = utils.keras_predict_batch(model, X_test, verbose=True)

    return prediction

# construct experiment cuts
exp_names = {
    "jul09/rec1499656391.hdf5": [2000, 4000],
    "jul09/rec1499657850.hdf5": [500, 800],
    "aug01/rec1501649676.hdf5": [500, 500],
    "aug01/rec1501650719.hdf5": [500, 500],
    "aug05/rec1501994881.hdf5": [200, 800],
    "aug09/rec1502336427.hdf5": [100, 400],
    "aug09/rec1502337436.hdf5": [100, 400],
    "jul16/rec1500220388.hdf5": [500, 200],
    "jul18/rec1500383971.hdf5": [500, 1000],
    "jul18/rec1500402142.hdf5": [200, 2000],
    "jul28/rec1501288723.hdf5": [200, 1000],
    "jul29/rec1501349894.hdf5": [200, 1500],
    "aug01/rec1501614399.hdf5": [200, 800],
    "aug08/rec1502241196.hdf5": [500, 1000],
    "aug15/rec1502825681.hdf5": [500, 1700]
}

# construct experiment names
exp_des = {
    "jul09/rec1499656391.hdf5": "night-1",
    "jul09/rec1499657850.hdf5": "night-2",
    "aug01/rec1501649676.hdf5": "night-3",
    "aug01/rec1501650719.hdf5": "night-4",
    "aug05/rec1501994881.hdf5": "night-5",
    "aug09/rec1502336427.hdf5": "night-6",
    "aug09/rec1502337436.hdf5": "night-7",
    "jul16/rec1500220388.hdf5": "day-1",
    "jul18/rec1500383971.hdf5": "day-2",
    "jul18/rec1500402142.hdf5": "day-3",
    "jul28/rec1501288723.hdf5": "day-4",
    "jul29/rec1501349894.hdf5": "day-5",
    "aug01/rec1501614399.hdf5": "day-6",
    "aug08/rec1502241196.hdf5": "day-7",
    "aug15/rec1502825681.hdf5": "day-8"
}

for exp in exp_des:
    exp_id = exp_des[exp]
    # load data
    data_path = os.path.join(
        spiker.SPIKER_EXPS, "ddd17", exp)
    frame_cut = exp_names[exp]
    # frame model base names
    model_base = "-"+exp_id+"-"
    sensor_type = ["full", "dvs", "aps"]

    print ("[MESSAGE] Data path:", data_path)
    print ("[MESSAGE] Frame cut:", frame_cut)

    # load ground truth
    test_frames, _ = ddd17.prepare_train_data(data_path,
                                              y_name="steering",
                                              frame_cut=frame_cut)
    test_frames /= 255.
    test_frames -= np.mean(test_frames, keepdims=True)
    num_samples = test_frames.shape[0]
    num_train = int(num_samples*0.7)
    X_test = test_frames[num_train:]
    del test_frames

    # loop over runs
    for run_idx in xrange(1, 5):
        steer_full = get_prediction(
            X_test, "steering", model_base,
            sensor_type[0], spiker.SPIKER_EXPS+"-run-%d" % (run_idx))
        steer_dvs = get_prediction(
            X_test[:, :, :, 0][..., np.newaxis], "steering", model_base,
            sensor_type[1], spiker.SPIKER_EXPS+"-run-%d" % (run_idx))
        steer_aps = get_prediction(
            X_test[:, :, :, 1][..., np.newaxis], "steering", model_base,
            sensor_type[2], spiker.SPIKER_EXPS+"-run-%d" % (run_idx))
        # save exported results
        result_file = os.path.join(
            spiker.SPIKER_EXTRA, "exported-results",
            "steering"+model_base+"run-%d.pkl" % (run_idx))
        with open(result_file, "w") as f:
            pickle.dump([steer_full, steer_dvs, steer_aps], f)
            f.close()
        print ("[MESSAGE] Saved results for %s run %d" % (exp_id, run_idx))
