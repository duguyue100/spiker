"""ResNets for Brake Pedal Status Prediction.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

from sacred import Experiment

import numpy as np
from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import spiker
from spiker import log
from spiker.models import resnet
from spiker.data import ddd17

exp = Experiment("ResNet - Brake - Experiment")

exp.add_config({
    "model_name": "",  # the model name
    "data_name": "",  # the data name
    "channel_id": 0,  # which channel to chose, 0: dvs, 1: aps, 2: both
    "stages": 0,  # number of stages
    "blocks": 0,  # number of blocks of each stage
    "filter_list": [],  # number of filters per stage
    "nb_epoch": 0,  # number of training epochs
    "batch_size": 0,  # batch size
    "frame_cut": [],  # cut frames
    })


@exp.automain
def resnet_exp(model_name, data_name, channel_id, stages, blocks, filter_list,
               nb_epoch, batch_size, frame_cut):
    """Perform ResNet experiment."""
    model_path = os.path.join(spiker.SPIKER_EXPS, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    else:
        raise ValueError("[MESSAGE] This experiment has been done before."
                         " Create a new config model if you need.")
    model_pic = os.path.join(model_path, model_name+"-model-pic.png")
    model_file_base = os.path.join(model_path, model_name)

    # print model info
    log.log("[MESSAGE] Model Name: %s" % (model_name))
    log.log("[MESSAGE] Number of epochs: %d" % (nb_epoch))
    log.log("[MESSAGE] Batch Size: %d" % (batch_size))
    log.log("[MESSAGE] Number of stages: %d" % (stages))
    log.log("[MESSAGE] Number of blocks: %d" % (blocks))

    # load data
    data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                             data_name)
    if not os.path.isfile(data_path):
        raise ValueError("This dataset does not exist at %s" % (data_path))
    log.log("[MESSAGE] Dataset %s" % (data_path))
    assert len(frame_cut) == 2
    log.log("[MESSAGE] Frame cut: first at %d, last at %d"
            % (frame_cut[0], -frame_cut[1]))
    frames, steering = ddd17.prepare_train_data(
        data_path, y_name="brake",
        frame_cut=frame_cut)
    frames /= 255.
    frames -= np.mean(frames, keepdims=True)
    num_samples = frames.shape[0]
    num_train = int(num_samples*0.7)
    X_train = frames[:num_train]
    Y_train = steering[:num_train]
    X_test = frames[num_train:]
    Y_test = steering[num_train:]

    del frames, steering

    if channel_id != 2:
        X_train = X_train[:, :, :, channel_id][..., np.newaxis]
        X_test = X_test[:, :, :, channel_id][..., np.newaxis]

    log.log("[MESSAGE] Number of samples %d" % (num_samples))
    log.log("[MESSAGE] Number of train samples %d" % (X_train.shape[0]))
    log.log("[MESSAGE] Number of test samples %d" % (X_test.shape[0]))

    # setup image shape
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # Build model
    model = resnet.resnet_builder(
        model_name=model_name, input_shape=input_shape,
        batch_size=batch_size,
        filter_list=filter_list, kernel_size=(3, 3),
        output_dim=1, stages=stages, blocks=blocks,
        bottleneck=False, network_type="binary")

    model.summary()
    plot_model(model, to_file=model_pic, show_shapes=True,
               show_layer_names=True)

    # configure optimizer
    #  def step_decay(epoch):
    #      "step decay callback."""
    #      if epoch >= 80 and epoch < 120:
    #          return float(0.01)
    #      elif epoch >= 120:
    #          return float(0.001)
    #      else:
    #          return float(0.1)

    #  sgd = optimizers.SGD(lr=0.0, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=["binary_accuracy"])
    print ("[MESSAGE] Model is compiled.")
    model_file = model_file_base + \
        "-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_binary_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    #  scheduler = LearningRateScheduler(step_decay)

    csv_his_log = os.path.join(model_path, "csv_history.log")
    csv_logger = CSVLogger(csv_his_log, append=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=1)

    callbacks_list = [checkpoint, csv_logger, early_stop]

    # training
    model.fit(
        x=X_train, y=Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list)
