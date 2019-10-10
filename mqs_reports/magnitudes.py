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


def fit_spectra(f, p_sig, p_noise, type,
                df_mute=1.05):
    fmin = 0.1
    fmax = 6.0
    if type == 'LF':
        fmax = 0.9
    if type == 'HF':
        fmin = 1.0

    mute_24 = [1.9, 3.4]
    bol_1Hz_mask = np.array(
        (np.array((f > fmin, f < fmax)).all(axis=0),
         np.array((f < 1. / df_mute,
                   f > df_mute)).any(axis=0),
         np.array((f < mute_24[0],
                   f > mute_24[1])).any(axis=0),
         np.array(p_sig > p_noise * 3.)
         )
        ).all(axis=0)

    if sum(bol_1Hz_mask) > 5:
        res = np.polyfit(f[bol_1Hz_mask],
                         10 * np.log10(p_sig[bol_1Hz_mask]),
                         deg=1)

        # plt.plot(f, 10*np.log10(p_noise), 'k')
        # plt.plot(f, 10*np.log10(p_sig), 'r')
        # plt.plot(f[bol_1Hz_mask], 10*np.log10(p_sig[bol_1Hz_mask]),
        #          'orange', lw=2)
        A0 = res[1]
        tstar = res[0]
        # plt.plot(f, A0 + f * tstar)
        # plt.show()
    else:
        A0 = None
        tstar = None

    return A0, tstar
