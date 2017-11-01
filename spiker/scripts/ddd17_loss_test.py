"""Plot DDD17+ loss figures.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

import spiker
from spiker.models import utils

# load log
csv_log_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-3-5",
    "csv_history.log")
csv_log = utils.parse_csv_log(csv_log_path)

ad_train_loss = csv_log["loss"]
ad_test_loss = csv_log["val_loss"]

csv_log_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-aps-3-5",
    "csv_history.log")
csv_log = utils.parse_csv_log(csv_log_path)

a_train_loss = csv_log["loss"]
a_test_loss = csv_log["val_loss"]

csv_log_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-dvs-3-5",
    "csv_history.log")
csv_log = utils.parse_csv_log(csv_log_path)

d_train_loss = csv_log["loss"]
d_test_loss = csv_log["val_loss"]

plt.figure(figsize=(10, 5))
plt.plot(ad_train_loss, "r", label="APS-DVS Combined")
plt.plot(a_train_loss, "g", label="APS Only")
plt.plot(d_train_loss, "blue", label="DVS Only")
plt.legend()
plt.grid()
plt.title("Training Loss Plot")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "train-loss.png"),
            dpi=200, format="png")

plt.figure(figsize=(10, 5))
plt.plot(ad_test_loss, "r", label="APS-DVS Combined")
plt.plot(a_test_loss, "g", label="APS Only")
plt.plot(d_test_loss, "blue", label="DVS Only")
plt.legend()
plt.grid()
plt.title("Testing Loss Plot")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(os.path.join(spiker.SPIKER_EXTRA, "test-loss.png"),
            dpi=200, format="png")
