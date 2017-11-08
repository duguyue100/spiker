# This is a Python template Makefile, do modification as you want
#
# Project: spiker
# Author: Yuhuang Hu
# Email : duguyue100@gmail.com

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

run:

# CVPR Experiments

experimental:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/experimental.json

night-1-full:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-full.json

pycaer-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/pycaer_test.py

test:
	PYTHONPATH=$(PYTHONPATH) python 

dvs128-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/dvs128_test.py

spiker-setup:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/__init__.py

ddd17-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_test.py

ddd17-load-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_load_test.py

ddd17-prediction-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_prediction_test.py

ddd17-loss-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_loss_test.py

ddd17-export-video:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_export_video.py

# Experiments

resnet-steering-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/resnet-steering-3-5.json

resnet-steering-dvs-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-dvs-3-5.json

resnet-steering-aps-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-aps-3-5.json

resnet_steering-hw-2:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/resnet-steering-hw-2-3-5.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-2-dvs-3-5.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-2-aps-3-5.json

resnet_steering-hw-up-1:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/resnet-steering-hw-up-1-3-5.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-up-1-dvs-3-5.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-up-1-aps-3-5.json

resnet-steering-hw-2-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/resnet-steering-hw-2-3-5.json

resnet-steering-hw-2-dvs-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-2-dvs-3-5.json

resnet-steering-hw-2-aps-3-5:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering_single_channel.py with ./spiker/exps/configs/resnet-steering-hw-2-aps-3-5.json

cleanall:
