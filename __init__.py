#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from os.path import dirname, realpath

for i in [sys.argv[0], __file__]:
    path = dirname(realpath(i))
    if path not in sys.path:
        sys.path.append(path)