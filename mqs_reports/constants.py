#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants used by various routines
:copyright:
    Simon Stähler (mail@simonstaehler.com), 2021
:license:
    None
"""
import json
from os import path

mydir = path.dirname(path.abspath(__file__))

# Magnitude constants
with open(path.join(mydir, 'data/magnitude_parameters.json'), 'r') as jsonfile:
    magnitude = json.load(jsonfile)

# Magnitude constants
with open(path.join(mydir, 'data/magnitude_exceptions.json'), 'r') as jsonfile:
    mag_exceptions = json.load(jsonfile)

# Seconds per day and Sol
SEC_PER_DAY_EARTH = 86400
SEC_PER_DAY_MARS = 88775.2440
