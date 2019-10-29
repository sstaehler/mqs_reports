#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

import numpy as np


def mb_P(amplitude_dB, distance_degree):
    amplitude_dB = 10 ** (amplitude_dB / 20.)
    mbP_tmp = np.log10(amplitude_dB) + 1.4 * np.log10(distance_degree) + 9.5
    mag = (mbP_tmp) + 0.1 + \
          1. / 3. * np.max((np.zeros_like(amplitude_dB), 4.5 - mbP_tmp),
                           axis=0)

    return mag


def mb_S(amplitude_dB, distance_degree):
    amplitude_dB = 10 ** (amplitude_dB / 20.)
    mb_S_tmp = np.log10(amplitude_dB) + 2.2 * np.log10(distance_degree) + 8.4
    mag = (mb_S_tmp) + 0.1 + \
          1. / 3. * np.max((np.zeros_like(amplitude_dB), 4.5 - (mb_S_tmp)),
                           axis=0)
    return mag


def M2_4(amplitude_dB, distance_degree):
    if amplitude_dB is None:
        return None
    else:
        # mag = np.log10(amplitude) - np.log10(4.78e-11) + \
        #       (np.log10(distance) - np.log10(30.)) * 1.2 + 3.
        amp_true = 10 ** (amplitude_dB / 20.)
        A0_est = 10 ** (0.7 * np.log10(amp_true)-3.)
        mag = (2. / 3.) * (np.log10(A0_est) +
                           1.5 * np.log10(distance_degree) + 9.4) + 1.9
        return mag


def MFB(amplitude_dB, distance_degree):
    if amplitude_dB is None:
        return None
    else:
        # A0 is given in dB
        amp_true = 10 ** (amplitude_dB / 20.)
        mag = (2. / 3.) * (np.log10(amp_true) +
                           1.1 * np.log10(distance_degree) + 9.8) + 1.9
        return mag


def lorenz(x, A, x0, xw):
    w = (x - x0) / (xw / 2.)
    return 10 * np.log10(1 / (1 + w ** 2)) + A


def lorenz_att(x, A0, x0, tstar, xw, ampfac):
    w = (x - x0) / (xw / 2.)
    stf_amp = 1 / (1 + (x / 3.) ** 2)
    # return 10 * np.log10(1 + ampfac / (1 + w ** 2) * stf_amp) \
    #        + A0 - tstar * x * np.pi / np.log(10.) * 10.
    return A0 + 10 * np.log10(
        (1 + ampfac / (1 + w ** 2)) * stf_amp
        * np.exp(- tstar * x * np.pi))


def _remove_singles(array):
    for ix in range(0, len(array) - 1):
        if array[ix] and not (array[ix - 1] or array[ix + 1]):
            array[ix] = False


def fit_peak_att(f, p):
    from scipy.optimize import curve_fit
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(lorenz_att, f, 10. * np.log10(p),
                           bounds=((-230, 2.25, 0.1, 0.05, 4.),
                                   (-160, 2.5, 10.0, 0.4, 100.)),
                           sigma=f * 10.,
                           p0=(-210, 2.4, 2., 0.25, 10.))
    return popt


def fit_peak(f, p):
    from scipy.optimize import curve_fit
    try:
        # noinspection PyTypeChecker
        popt, pcov = curve_fit(lorenz, f, 10 * np.log10(p),
                               bounds=((-240, 2.3, 0.2),
                                       (-180, 2.5, 0.4)),
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


def fit_spectra(f, p_sig, p_noise, event_type, df_mute=1.05):
    if len(p_sig) < len(f):
        fac = len(f) // len(p_sig)
        f_dec = f[::fac]
        p_sig = np.interp(x=f, xp=f_dec, fp=p_sig)
    if len(p_noise) < len(f):
        fac = len(f) // len(p_noise)
        f_dec = f[::fac]
        p_sig = np.interp(x=f, xp=f_dec, fp=p_noise)
    fmin = 0.1
    fmax = 6.0
    if event_type in ['LF', 'BB']:
        fmax = 0.9
    if event_type == 'HF':
        fmin = 1.0

    mute_24 = [1.9, 3.4]
    bol_1Hz_mask = np.array(
        (np.array((f > fmin, f < fmax)).all(axis=0),
         np.array((f < 1. / df_mute,
                   f > df_mute)).any(axis=0),
         # np.array((f < mute_24[0],
         #           f > mute_24[1])).any(axis=0),
         np.array(p_sig > p_noise * 4.)
         )
        ).all(axis=0)
    _remove_singles(bol_1Hz_mask)
    A0 = None
    tstar = None
    ampfac = None
    width_24 = None
    f_24 = None
    A_24 = None
    if event_type is not '24':
        if sum(bol_1Hz_mask) > 5:
            if event_type in ['HF', 'VF']:
                try:
                    A0, f_24, tstar, width_24, ampfac = fit_peak_att(
                        f[bol_1Hz_mask],
                        p_sig[bol_1Hz_mask])
                except RuntimeError:
                    pass
                # plt.plot(f, 10 * np.log10(p_noise), 'k')
            else:
                res = np.polyfit(f[bol_1Hz_mask],
                                 10 * np.log10(p_sig[bol_1Hz_mask]),
                                 deg=1)
                A0 = res[1]
                tstar = - res[0] / 10. * np.log(10) / np.pi  # Because dB

    if event_type not in ['LF', 'BB']:
        bol_24_mask = np.array((f > mute_24[0],
                                f < mute_24[1])).all(axis=0)
        A_24, f_24, tmp = fit_peak(f[bol_24_mask], p_sig[bol_24_mask])
        if width_24 is None:
            width_24 = tmp

    amps = dict()
    amps['A0'] = A0
    amps['tstar'] = tstar
    amps['A_24'] = A_24
    amps['f_24'] = f_24
    amps['width_24'] = width_24
    amps['ampfac'] = ampfac
    return amps
