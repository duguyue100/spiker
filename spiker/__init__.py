"""Initialize Spiker.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function

import os

from spiker import utils

# setup basic environment
HOME = os.environ["HOME"]
SPIKER_ROOT = os.path.join(HOME, "spikeres")
SPIKER_DATA = os.path.join(SPIKER_ROOT, "data")
SPIKER_EXPS = os.path.join(SPIKER_ROOT, "exps")
SPIKER_CONFIG = os.path.join(SPIKER_ROOT, "config.json")

utils.makedir(SPIKER_ROOT, "Spiker root directory is at %s." % (SPIKER_ROOT))
utils.makedir(SPIKER_DATA, "Spiker data directory is at %s." % (SPIKER_DATA))
utils.makedir(SPIKER_EXPS,
              "Spiker experiments directory is at %s." % (SPIKER_EXPS))
utils.makefile(SPIKER_CONFIG,
               "Spiker configuration is at %s." % (SPIKER_CONFIG))
