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
    amplitude = 10 ** (amplitude_dB / 20.)
    mag = 0.7318 * np.log10(amplitude) + 1.2 * np.log10(distance_degree)+ 8.3471

    return mag


def mb_S(amplitude_dB, distance_degree):
    amplitude = 10 ** (amplitude_dB / 20.)
    mag = 0.7647 * np.log10(amplitude)+ 1.4 * np.log10(distance_degree) + 8.0755

    return mag


def M2_4(amplitude_dB, distance_degree):
    if amplitude_dB is None:
        return None
    else:
        amplitude = 10 ** (amplitude_dB / 20.)
        mag = 0.6177 * np.log10(amplitude) + 0.9 * np.log10(distance_degree) + \
              7.0026
        return mag

def MFB(amplitude_dB, distance_degree):
    dist_term = 1.1
    if amplitude_dB is None:
        return None
    else:
        logM0 = amplitude_dB / 20. + \
                dist_term * np.log10(distance_degree) + \
                21.475
        mag = 2. / 3. * (logM0 - 9.1)
        return mag


def MFB_HF(amplitude_dB, distance_degree):
    dist_term = 0.9
    if amplitude_dB is None:
        return None
    else:
        logM0 = amplitude_dB / 20. + \
                dist_term * np.log10(distance_degree) + \
                21.475
        mag = 2. / 3. * (logM0 - 9.1)
        return mag


def lorenz(x, A, x0, xw):
    w = (x - x0) / (xw / 2.)
    return 10 * np.log10(1 / (1 + w ** 2)) + A


def lorenz_att(f: np.array,
               A0: float,
               f0: float,
               f_c: float,
               tstar: float,
               fw: float,
               ampfac: float):
    """
    Attenuation spectrum, combined with Lorenz peak and source spectrum
    :param f: Frequency array (in Hz)
    :param A0: Long-period amplitude in flat part of spectrum (in dB)
    :param f0: Center frequency of the Lorenz peak (aka 2.4 Hz)
    :param f_c: Corner frequency of the Source (Hz)
    :param tstar: T* value from attenuation
    :param fw: Width of Lorenz peak
    :param ampfac: Amplification factor of Lorenz peak
    :return:
    """
    w = (f - f0) / (fw / 2.)
    stf_amp = 1 / (1 + (f / f_c) ** 2)
    return A0 + 10 * np.log10(
        (1 + ampfac / (1 + w ** 2))
        * stf_amp
        * np.exp(- tstar * f * np.pi))


def _remove_singles(array):
    for ix in range(0, len(array) - 1):
        if array[ix] and not (array[ix - 1] or array[ix + 1]):
            array[ix] = False


def fit_peak_att(f, p, A0_max=-200, tstar_min=0.05):
    from scipy.optimize import curve_fit
    tstar_max = 10.0

    # Central frequency of the 2.4 Hz mode (in Hz)
    f0_min = 2.25
    f0_max = 2.5

    # Amplification factor of the 2.4 Hz mode (not in dB!)
    ampfac_min = 10.
    ampfac_max = 400.

    # Width of the 2.4 Hz mode
    fw_min = 0.05
    fw_max = 0.4
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(lorenz_att, f, 10. * np.log10(p),
                           bounds=(
                               (-240, 
                                f0_min,
                                0.8, 
                                tstar_min,
                                fw_min,
                                ampfac_min),
                               (A0_max, 
                                f0_max,
                                10.0, 
                                tstar_max,
                                fw_max,
                                ampfac_max)),
                           sigma=f * 10.,
                           p0=(A0_max - 5, 2.4, 3., 2., 0.25, ampfac_min))
    return popt


def fit_peak(f, p):
    from scipy.optimize import curve_fit
    # Central frequency of the 2.4 Hz mode (in Hz)
    f0_min = 2.25
    f0_max = 2.5
    # Width of the 2.4 Hz mode
    fw_min = 0.05
    fw_max = 0.4
    try:
        # noinspection PyTypeChecker
        popt, pcov = curve_fit(lorenz, f, 10 * np.log10(p),
                               bounds=((-240, f0_min, fw_min),
                                       (-180, f0_max, fw_max)),
                               p0=(-210, 2.4, 0.25))
    except ValueError:
        popt = [-250, 2.4, 1.0]

    return popt


def fit_spectra(f, p_sig, p_noise, event_type, df_mute=1.05):
    if len(p_sig) < len(f):
        f_dec = np.linspace(f[0], f[-1], len(p_sig)) 
        p_sig = np.interp(x=f, xp=f_dec, fp=p_sig)
    if len(p_noise) < len(f):
        f_dec = np.linspace(f[0], f[-1], len(p_noise)) 
        p_sig = np.interp(x=f, xp=f_dec, fp=p_noise)
    fmin = 0.1
    fmax = 6.0
    if event_type in ['LF', 'BB']:
        fmax = 0.9
    if event_type == 'HF':
        fmin = 1.0
    if event_type == 'UF':
        fmin = 4.0
        fmax = 9.0

    mute_24 = [1.9, 3.4]
    bol_1Hz_mask = np.array(
        (np.array((f > fmin, f < fmax)).all(axis=0),
         np.array((f < 1. / df_mute,
                   f > df_mute)).any(axis=0),
         np.array(p_sig > p_noise * 2.)
         )
        ).all(axis=0)
    _remove_singles(bol_1Hz_mask)
    A0 = None
    tstar = None
    ampfac = None
    width_24 = None
    f_24 = None
    f_c = None
    A_24 = None
    if event_type is not '24':
        if sum(bol_1Hz_mask) > 5:

            if event_type in ['HF', 'VF']:
                # A0 should not be larger than peak between 1.1 and 1.8 Hz
                A0_max = np.max(10 * np.log10(
                    p_sig[np.array((f > 1.1, f < 1.8)).all(axis=0)])) + 6.
                # tstar must be so large than event is below noise
                if max(f[bol_1Hz_mask]) < 4:
                    ifreq = np.array((f > 6.0, f < 7.0)).all(axis=0)
                    tstar_min = (np.log(10 ** (A0_max / 10)) -
                                 np.log(np.mean(p_sig[ifreq]))) \
                                / (np.pi * 6.5)
                else:
                    tstar_min = 0.05
                try:
                    A0, f_24, f_c, tstar, width_24, ampfac = fit_peak_att(
                        f[bol_1Hz_mask],
                        p_sig[bol_1Hz_mask],
                        A0_max, tstar_min)
                except RuntimeError:
                    pass
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
    amps['f_c'] = f_c
    amps['width_24'] = width_24
    amps['ampfac'] = ampfac
    return amps
