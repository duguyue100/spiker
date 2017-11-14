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

multi-run-exps:
	# aps night
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-aps.json
	# aps day
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-aps.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-aps.json
	# dvs night
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-dvs.json
	# dvs day
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-dvs.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-dvs.json
	# full night
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-full.json
	# full day
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-full.json
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-full.json

rerun-exps:
	# APS
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-aps.json
	# DVS
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-dvs.json
	# FULL
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-full.json

night-1234567-aps:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-1-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-1-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-3-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-3-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-4-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-4-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-5-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-5-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-6-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-6-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-7-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-7-aps.json

day-12345678-aps:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-1-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-1-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-3-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-3-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-4-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-4-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-5-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-5-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-6-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-6-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-aps.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-aps.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-aps.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-aps.json

night-1234567-dvs:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-1-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-1-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-3-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-3-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-4-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-4-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-5-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-5-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-6-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-6-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-7-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-7-dvs.json

day-12345678-dvs:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-1-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-1-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-3-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-3-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-4-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-4-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-5-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-5-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-6-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-6-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-dvs.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-dvs.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-dvs.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-dvs.json

night-1234567-full:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-1-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-1-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-1-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-2-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-2-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-2-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-3-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-3-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-3-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-4-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-4-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-4-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-5-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-5-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-5-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-6-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-6-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-6-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-night-7-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-night-7-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-night-7-full.json

day-12345678-full:
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-1-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-1-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-1-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-2-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-2-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-2-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-3-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-3-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-3-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-4-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-4-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-4-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-5-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-5-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-5-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-6-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-6-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-6-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-7-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-7-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-7-full.json
	# steering
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_steering.py with ./spiker/exps/configs/cvprexps/steering-day-8-full.json
	# accel
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_accel.py with ./spiker/exps/configs/cvprexps/accel-day-8-full.json
	# brake
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./spiker/exps/resnet_brake.py with ./spiker/exps/configs/cvprexps/brake-day-8-full.json

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

ddd17-create-configs:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_create_configs.py

cvpr-create-figures:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/cvpr_create_figures.py

ddd17-fields:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_export_fields.py

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
