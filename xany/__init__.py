#! /usr/bin/env python
# -*- coding: utf-8 -*-


from . import prep, utils, process, stitch
import warnings

__version_info__ = ('0', '7', '0')
__version__ = '.'.join(__version_info__)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()