#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import numpy as np


def mb_P(amplitude, distance):
    mag = (np.log10(amplitude) + 1.4 * np.log10(distance) + 9.5) + 0.1 + \
          1. / 3. * max(0, 4.5 - (np.log10(amplitude)
                                  + 1.4 * np.log10(distance) + 9.5))

    return mag


def mb_S(amplitude, distance):
    mag = (np.log10(amplitude) + 2.2 * np.log10(distance) + 8.4) + 0.1 + \
          1. / 3. * max(0, 4.5 - (np.log10(amplitude)
                                  + 2.2 * np.log10(distance) + 8.4))
    return mag


def M2_4(amplitude, distance):
    if amplitude is None:
        return None
    else:
        mag = np.log10(amplitude) - np.log10(4.78e-11) + \
              (np.log10(distance) - np.log10(30.)) * 1.2 + 3.
        return mag
