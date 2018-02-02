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

pycaer-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/pycaer_test.py

dvs128-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/dvs128_test.py

spiker-setup:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/__init__.py

install:
	python setup.py install

ddd17-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/ddd17_test.py

rosbag-exporter:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/rosbag_exporter.py

rosbag-new-bind-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/rosbag_bind_driver_test.py

hdf5-test:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/hdf5_test.py

hdf5-exporter:
	PYTHONPATH=$(PYTHONPATH) python ./spiker/scripts/hdf5_exporter.py
