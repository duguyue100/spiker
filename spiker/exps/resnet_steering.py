"""ResNets for Steering Prediction.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

from sacred import Experiment

from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import spiker
from spiker import log
from spiker.models import resnet

exp = Experiment("ResNet - Steering - Experiment")

exp.add_config({
    "model_name": "",  # the model name
    "stages": 0,  # number of stages
    "blocks": 0,  # number of blocks of each stage
    "filter_list": [],  # number of filters per stage
    "nb_epoch": 0,  # number of training epochs
    "batch_size": 0,  # batch size
    })


@exp.automain
def resnet_exp(model_name, stages, blocks, filter_list,
               nb_epoch, batch_size):
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

    # setup image shape
    input_shape = (64, 64, 2)

    # Build model
    model = resnet.resnet_builder(
        model_name=model_name, input_shape=input_shape,
        filter_list=filter_list, kernel_size=(3, 3),
        last_dim=16, output_dim=1, stages=stages, blocks=blocks,
        bottleneck=False)

    model.summary()
    plot_model(model, to_file=model_pic, show_shapes=True,
               show_layer_names=True)

    # configure optimizer
    def step_decay(epoch):
        "step decay callback."""
        if epoch >= 80 and epoch < 120:
            return float(0.01)
        elif epoch >= 120:
            return float(0.001)
        else:
            return float(0.1)

    sgd = optimizers.SGD(lr=0.0, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=["mse"])
    print ("[MESSAGE] Model is compiled.")
    model_file = model_file_base + \
        "-{epoch:02d}-{val_mean_squared_error:.2f}.hdf5"
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    scheduler = LearningRateScheduler(step_decay)

    csv_his_log = os.path.join(model_path, "csv_history.log")
    csv_logger = CSVLogger(csv_his_log, append=True)

    callbacks_list = [checkpoint, scheduler, csv_logger]

    # configure data stream
    datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=False,
        vertical_flip=False,
        data_format="channels_last")

    # training
    model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size),
            steps_per_epoch=X_train.shape[0] // batch_size,
            epochs=nb_epoch, validation_data=(X_test, Y_test),
            callbacks=callbacks_list)