#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import numpy as np
from mqs_reports.event import Event


def calc_SNR(event: Event, fmin, fmax):
    p_noise = event.spectra['noise']['p_Z']
    df_noise = event.spectra['noise']['f'][1]
    f_bool = np.array((event.spectra['noise']['f'] > fmin,
                       event.spectra['noise']['f'] < fmax)).all(axis=0)
    power_noise = np.trapz(p_noise[f_bool], dx=df_noise)

    for type in ['S', 'P', 'all']:
        if type in event.spectra:
            p_signal = event.spectra[type]['p_Z']
            df_signal = event.spectra[type]['f'][1]
            f_bool = np.array((event.spectra[type]['f'] > fmin,
                               event.spectra[type]['f'] < fmax)).all(axis=0)
            continue

    power_signal = np.trapz(p_signal[f_bool], dx=df_signal)
    return power_signal / power_noise
