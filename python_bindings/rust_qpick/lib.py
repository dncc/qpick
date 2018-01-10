import os
import re
import sys
from ._ffi import ffi

def find_library():
    libname = "qpickwrapper"
    if sys.platform == 'win32':
        prefix = ''
        suffix = 'dll'
    elif sys.platform == 'darwin':
        prefix = 'lib'
        suffix = 'dylib'
    else:
        prefix = 'lib'
        suffix = 'so'
    cur_dir = os.path.dirname(__file__)
    return os.path.join(cur_dir, "{}{}.{}".format(prefix, libname, suffix))

lib = ffi.dlopen(find_library())
