#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

import numpy as np

from mqs_reports.utils import linregression


def mb_P(amplitude_dB, distance_degree):
    amplitude = 10 ** (amplitude_dB / 20.)
    mag = 0.7318 * np.log10(amplitude) + 1.2 * np.log10(
        distance_degree) + 8.3471

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


def lorentz(x, A, x0, xw):
    """
    Return a Lorentz peak function centered around x0, with width xw and
    amplitude A in dB values

    Parameters
    ----------
    :param x: x-values to evaluate function at
    :param A: Peak amplitude
    :param x0: center value of peak
    :param xw: width of peak
    :return: Lorentz/Cauchy function in dB
    """
    w = (x - x0) / (xw / 2.)
    return 10. * np.log10(1. / (1. + w ** 2)) + A

def lorentz_modes(x, A, x0, xw, ampfac):
    """
    Return a Lorentz peak function centered around x0, with width xw and
    amplitude A in dB values

    Parameters
    ----------
    :param x: x-values to evaluate function at
    :param A: Peak amplitude
    :param x0: center value of peak
    :param xw: width of peak
    :return: Lorentz/Cauchy function in dB
    """
    w = (x - x0) / (xw / 2.)
    return A + 10 * np.log10(1 + ampfac / (1 + w ** 2))
    # return A * (1 + ampfac / (1 + w ** 2))

def lorentz_att(f: np.array,
                A0: float,
                f0: float,
                f_c: float,
                tstar: float,
                fw: float,
                ampfac: float):
    """
    Attenuation spectrum, combined with Lorentz peak and source spectrum
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
    popt, pcov = curve_fit(lorentz_att, f, 10. * np.log10(p),
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


def fit_peak(f, p, A0_min=-240, A0_max=-180,
             f0_min=2.25, f0_max=2.5, fw_min=0.05, fw_max=0.4):
    """
    Fit a spectral peak to function PSD p at frequencies f
    Parameters
    ----------
    :param f: frequency vector [in Hz]
    :param p: power spectral density [in dB]
    :param A0_max: Minimum allowed amplitude for peak
    :param A0_min: Maximum allowed amplitude for peak
    :param f0_min: Minimum allowed frequency for peak
    :param f0_max: Maximum allowed frequency for peak
    :param fw_min: Minimum allowed spectral width [in Hz]
    :param fw_max: Maximum allowed spectral width [in Hz]
    :return: list with Amplitude, central frequency, width of peak
    """
    from scipy.optimize import curve_fit
    # try:
        # noinspection PyTypeChecker
        # popt, pcov = curve_fit(lorentz, f, 10 * np.log10(p),
        #                        bounds=((A0_min, f0_min, fw_min),
        #                                (A0_max, f0_max, fw_max)),
        #                        p0=((A0_max + A0_min) * 0.5,
        #                            (f0_max + f0_min * 0.5),
        #                            (fw_max + fw_min) * 0.5))
    popt, pcov = curve_fit(lorentz, f, p,
                           bounds=((A0_min, f0_min, fw_min),
                                   (A0_max, f0_max, fw_max)),
                           p0=((A0_max + A0_min) * 0.5,
                               (f0_max + f0_min * 0.5),
                               (fw_max + fw_min) * 0.5))    
    # except ValueError:
    #     popt = [-250, 2.4, 1.0]

    return popt

def fit_peak_modes(f, p, A0_min=-250, A0_max=-135,
                   f0_min=2.25, f0_max=2.5, fw_min=0.05, fw_max=0.4, 
                   ampfac_min = 100., ampfac_max = 400.): #10, 400
    """
    Fit a spectral peak to function PSD p at frequencies f

    Parameters
    ----------
    :param f: frequency vector [in Hz]
    :param p: power spectral density [in dB]
    :param A0_max: Minimum allowed amplitude for peak
    :param A0_min: Maximum allowed amplitude for peak
    :param f0_min: Minimum allowed frequency for peak
    :param f0_max: Maximum allowed frequency for peak
    :param fw_min: Minimum allowed spectral width [in Hz]
    :param fw_max: Maximum allowed spectral width [in Hz]
    ampfac_min/max: Amplification factor of the mode (not in dB!)
    
    :return: list with Amplitude, central frequency, width of peak
    """
    from scipy.optimize import curve_fit
        # noinspection PyTypeChecker
    popt, pcov = curve_fit(lorentz_modes, f, p,
                           bounds=((A0_min, f0_min, fw_min, ampfac_min),
                                   (A0_max, f0_max, fw_max, ampfac_max)),
                           p0=((A0_max + A0_min) * 0.5,
                               (f0_max + f0_min) * 0.5,
                               (fw_max + fw_min) * 0.5,
                               (ampfac_max + ampfac_min) * 0.5))

    return popt

def fit_spectra(f_sig, p_sig, f_noise, p_noise, event_type, df_mute=1.05):
    len_spec = len(f_noise)
    if len(p_sig) != len_spec:
        f_dec = np.linspace(f_noise[0], f_noise[-1], len_spec)
        p_sig = np.interp(x=f_noise, xp=f_sig, fp=p_sig)
    # if len(p_noise) != len_spec:
    #     f_dec = np.linspace(f[0], f[-1], len_spec)
    #     p_noise = np.interp(x=f, xp=f_dec, fp=p_noise)
    f = f_noise
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
    A0_err = None
    tstar_err = None
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
                # res = np.polyfit(f[bol_1Hz_mask],
                #                  10 * np.log10(p_sig[bol_1Hz_mask]),
                #                  deg=1)
                res = linregression(x=f[bol_1Hz_mask],
                                    y=10 * np.log10(p_sig[bol_1Hz_mask]),
                                    q=0.95)
                # A0 = res[1]
                # tstar = - res[0] / 10. * np.log(10) / np.pi  # Because dB
                A0 = res[0]
                tstar = - res[2] / 10. * np.log(10) / np.pi  # Because dB
                A0_err = res[1]
                tstar_err = - res[3] / 10. * np.log(10) / np.pi  # Because dB

    if event_type not in ['LF', 'BB']:
        bol_24_mask = np.array((f > mute_24[0],
                                f < mute_24[1])).all(axis=0)
        A_24, f_24, tmp = fit_peak(f[bol_24_mask], p_sig[bol_24_mask])
        if width_24 is None:
            width_24 = tmp

    amps = dict()
    amps['A0'] = A0
    amps['tstar'] = tstar
    amps['A0_err'] = A0_err
    amps['tstar_err'] = tstar_err
    amps['A_24'] = A_24
    amps['f_24'] = f_24
    amps['f_c'] = f_c
    amps['width_24'] = width_24
    amps['ampfac'] = ampfac
    return amps


def fit_spectra_modes(f_sig, p_sig, mute_24, fminmax, width_peak, ampFactor):
    import matplotlib.pyplot as plt
    
    f = f_sig

    mute_24 = mute_24

    width_24 = None
    f_24 = None
    A_24 = None

    bol_24_mask = np.array((f > mute_24[0],
                            f < mute_24[1])).all(axis=0)
    
    # #debug plot part1
    # plt.plot(f, p_sig)
    # plt.plot(f[bol_24_mask], p_sig[bol_24_mask])
    
    # A_24, f_24, tmp, ampfac_24 = fit_peak(f[bol_24_mask], p_sig[bol_24_mask],
    #                                 f0_min = fminmax[0], f0_max = fminmax[-1],
    #                                 fw_min = width_peak[0], fw_max = width_peak[-1])
    
    A_24, f_24, width_24, ampfac_24 = fit_peak_modes(f[bol_24_mask], p_sig[bol_24_mask],
                                    f0_min = fminmax[0], f0_max = fminmax[-1],
                                    fw_min = width_peak[0], fw_max = width_peak[-1],
                                    ampfac_min = ampFactor[0], ampfac_max=ampFactor[-1]) #50,550 works

    # #debug plot part2
    # plt.plot(f[bol_24_mask],lorentz_modes(x=f[bol_24_mask],A=A_24, x0=f_24, xw=width_24, ampfac=ampfac_24))
    # plt.ylim(-250,-150)
    # plt.text(x=1, y=-180, s=f'{A_24:6.1f}dB {f_24:6.3f}Hz {width_24:6.4f}Hz {ampfac_24:6.1f}')
    # plt.show()
    
    if ampfac_24  < 10.0: #If amplitude is too small -> Mode not properly detected
        f_24 = None
        A_24 = None
        width_24 = None
        ampfac_24 = None


    amps = dict()
    amps['A_24'] = A_24
    amps['f_24'] = f_24
    amps['width_24'] = width_24
    amps['ampfac'] = ampfac_24
    return amps
