import os
import sys
from easydict import EasyDict

CONF = EasyDict()
# scalar
CONF.SCALAR = EasyDict()
CONF.SCALAR.OBJ_PC_SAMPLE = 1000
CONF.SCALAR.REL_PC_SAMPLE = 3000

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "E:/wff_ftp_server/3DSGG"
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data/3RScan")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# 3RScan data
CONF.PATH.R3Scan = os.path.join(CONF.PATH.DATA, "3RScan")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")