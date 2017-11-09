"""DDD17+ create configs.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os
from os.path import join

import json

import spiker

# configure path
config_path = os.path.join(
    spiker.HOME, "workspace", "spiker", "spiker", "exps", "configs",
    "cvprexps")

print ("hi, I'm here")

# for day, for DVS
for idx in range(1, 9):
    source_base = "-day-%d-full.json" % (idx)
    dvs_source_base = "-day-%d-dvs.json" % (idx)
    steer_path = os.path.join(config_path, "steering"+source_base)
    accel_path = os.path.join(config_path, "accel"+source_base)
    brake_path = os.path.join(config_path, "brake"+source_base)

    # open files
    with open(steer_path, "r") as f:
        steer_json = json.load(f)
        f.close()
    with open(accel_path, "r") as f:
        accel_json = json.load(f)
        f.close()
    with open(brake_path, "r") as f:
        brake_json = json.load(f)
        f.close()

    # modify fields
    steer_json["model_name"] = "steering-day-%d-dvs" % (idx)
    steer_json["channel_id"] = 0
    accel_json["model_name"] = "accel-day-%d-dvs" % (idx)
    accel_json["channel_id"] = 0
    brake_json["model_name"] = "brake-day-%d-dvs" % (idx)
    brake_json["channel_id"] = 0

    # write to file
    with open(join(config_path, "steering"+dvs_source_base), "w") as f:
        json.dump(steer_json, f)
        f.close()
    with open(join(config_path, "accel"+dvs_source_base), "w") as f:
        json.dump(accel_json, f)
        f.close()
    with open(join(config_path, "brake"+dvs_source_base), "w") as f:
        json.dump(brake_json, f)
        f.close()

# for day, for APS
for idx in range(1, 9):
    source_base = "-day-%d-full.json" % (idx)
    aps_source_base = "-day-%d-aps.json" % (idx)
    steer_path = os.path.join(config_path, "steering"+source_base)
    accel_path = os.path.join(config_path, "accel"+source_base)
    brake_path = os.path.join(config_path, "brake"+source_base)

    # open files
    with open(steer_path, "r") as f:
        steer_json = json.load(f)
        f.close()
    with open(accel_path, "r") as f:
        accel_json = json.load(f)
        f.close()
    with open(brake_path, "r") as f:
        brake_json = json.load(f)
        f.close()

    # modify fields
    steer_json["model_name"] = "steering-day-%d-aps" % (idx)
    steer_json["channel_id"] = 1
    accel_json["model_name"] = "accel-day-%d-aps" % (idx)
    accel_json["channel_id"] = 1
    brake_json["model_name"] = "brake-day-%d-aps" % (idx)
    brake_json["channel_id"] = 1

    # write to file
    with open(join(config_path, "steering"+aps_source_base), "w") as f:
        json.dump(steer_json, f)
        f.close()
    with open(join(config_path, "accel"+aps_source_base), "w") as f:
        json.dump(accel_json, f)
        f.close()
    with open(join(config_path, "brake"+aps_source_base), "w") as f:
        json.dump(brake_json, f)
        f.close()

# for night, for DVS
for idx in range(1, 8):
    source_base = "-night-%d-full.json" % (idx)
    dvs_source_base = "-night-%d-dvs.json" % (idx)
    steer_path = os.path.join(config_path, "steering"+source_base)
    accel_path = os.path.join(config_path, "accel"+source_base)
    brake_path = os.path.join(config_path, "brake"+source_base)

    # open files
    with open(steer_path, "r") as f:
        steer_json = json.load(f)
        f.close()
    with open(accel_path, "r") as f:
        accel_json = json.load(f)
        f.close()
    with open(brake_path, "r") as f:
        brake_json = json.load(f)
        f.close()

    # modify fields
    steer_json["model_name"] = "steering-night-%d-dvs" % (idx)
    steer_json["channel_id"] = 0
    accel_json["model_name"] = "accel-night-%d-dvs" % (idx)
    accel_json["channel_id"] = 0
    brake_json["model_name"] = "brake-night-%d-dvs" % (idx)
    brake_json["channel_id"] = 0

    # write to file
    with open(join(config_path, "steering"+dvs_source_base), "w") as f:
        json.dump(steer_json, f)
        f.close()
    with open(join(config_path, "accel"+dvs_source_base), "w") as f:
        json.dump(accel_json, f)
        f.close()
    with open(join(config_path, "brake"+dvs_source_base), "w") as f:
        json.dump(brake_json, f)
        f.close()

# for night, for APS
for idx in range(1, 8):
    source_base = "-night-%d-full.json" % (idx)
    aps_source_base = "-night-%d-aps.json" % (idx)
    steer_path = os.path.join(config_path, "steering"+source_base)
    accel_path = os.path.join(config_path, "accel"+source_base)
    brake_path = os.path.join(config_path, "brake"+source_base)

    # open files
    with open(steer_path, "r") as f:
        steer_json = json.load(f)
        f.close()
    with open(accel_path, "r") as f:
        accel_json = json.load(f)
        f.close()
    with open(brake_path, "r") as f:
        brake_json = json.load(f)
        f.close()

    # modify fields
    steer_json["model_name"] = "steering-night-%d-aps" % (idx)
    steer_json["channel_id"] = 1
    accel_json["model_name"] = "accel-night-%d-aps" % (idx)
    accel_json["channel_id"] = 1
    brake_json["model_name"] = "brake-night-%d-aps" % (idx)
    brake_json["channel_id"] = 1

    # write to file
    with open(join(config_path, "steering"+aps_source_base), "w") as f:
        json.dump(steer_json, f)
        f.close()
    with open(join(config_path, "accel"+aps_source_base), "w") as f:
        json.dump(accel_json, f)
        f.close()
    with open(join(config_path, "brake"+aps_source_base), "w") as f:
        json.dump(brake_json, f)
        f.close()
