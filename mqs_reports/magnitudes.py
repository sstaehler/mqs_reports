#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import matplotlib.pyplot as plt
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
        # mag = np.log10(amplitude) - np.log10(4.78e-11) + \
        #       (np.log10(distance) - np.log10(30.)) * 1.2 + 3.
        amp_true = 10 ** (amplitude / 20.)
        mag = (2. / 3.) * (np.log10(amp_true) +
                           1.0 * np.log10(distance) + 9.8) + 1.8
        return mag


def MFB(amplitude, distance):
    if amplitude is None:
        return None
    else:
        # A0 is given in dB
        amp_true = 10 ** (amplitude / 20.)
        mag = (2. / 3.) * (np.log10(amp_true) +
                           1.1 * np.log10(distance) + 9.8) + 1.9
        return mag


def lorenz(x, A, x0, xw):
    w = (x - x0) / (xw / 2.)
    return 10 * np.log10(1 / (1 + w ** 2)) + A


def fit_peak(f, p):
    from scipy.optimize import curve_fit
    try:
        popt, pcov = curve_fit(lorenz, f, 10 * np.log10(p),
                               bounds=((-240, 2.3, 0.2), (-180, 2.5, 0.4)),
                               p0=(-210, 2.4, 0.25))
    except ValueError:
        popt = [-250, 2.4, 1.0]

    # plt.plot(f, 10 * np.log10(p), 'b')
    # plt.plot(f, lorenz(f, *popt), 'r')
    # plt.show()
    # def func(x, A, A0):
    #     x0 = 2.4
    #     xw = 0.4
    #     w = (x-x0) / (xw / 2.)
    #     return A / (1 + w**2) + A0

    # popt, pcov = curve_fit(func, f, p, bounds=(0, 1), p0=(1e-20, 1e-22))

    # plt.plot(f, 10*np.log10(p), 'b')
    # plt.plot(f, 10*np.log10(func(f, popt[0], popt[1])), 'r')
    # plt.show()
    return popt


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
    A0 = None
    tstar = None
    if type is not '24':
        if sum(bol_1Hz_mask) > 5:
            res = np.polyfit(f[bol_1Hz_mask],
                             10 * np.log10(p_sig[bol_1Hz_mask]),
                             deg=1)

            plt.plot(f, 10 * np.log10(p_noise), 'k')
            # plt.plot(f, 10*np.log10(p_sig), 'r')
            # plt.plot(f[bol_1Hz_mask], 10*np.log10(p_sig[bol_1Hz_mask]),
            #          'orange', lw=2)
            A0 = res[1]
            tstar = res[0]
            # plt.plot(f, A0 + f * tstar)
            # plt.show()

    if type is not 'LF':
        bol_24_mask = np.array((f > mute_24[0],
                                f < mute_24[1])).all(axis=0)
        A_24, f_24, width_24 = fit_peak(f[bol_24_mask], p_sig[bol_24_mask])
    else:
        A_24 = None
        f_24 = None
        width_24 = None

    amps = dict()
    amps['A0'] = A0
    amps['tstar'] = tstar
    amps['A_24'] = A_24
    amps['f_24'] = f_24
    amps['width_24'] = width_24
    return amps
