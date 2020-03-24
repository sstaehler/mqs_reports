#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from typing import Tuple

import numpy as np

from mqs_reports.event import Event


def calc_SNR(event: Event, fmin: float, fmax: float,
             hor=False, SP=False) -> Tuple[float, str]:
    if SP:
        spectra = event.spectra_SP
    else:
        spectra = event.spectra
    if hor:
        comp = 'p_H'
    else:
        comp = 'p_Z'
    p_noise = spectra['noise'][comp]
    df_noise = spectra['noise']['f'][1]
    f_bool = np.array((spectra['noise']['f'] > fmin,
                       spectra['noise']['f'] < fmax)).all(axis=0)
    power_noise = np.trapz(p_noise[f_bool], dx=df_noise)
    for spec_win in ['S', 'P', 'all']:
        if spec_win in spectra:
            p_signal = spectra[spec_win][comp]
            df_signal = spectra[spec_win]['f'][1]
            f_bool = np.array((spectra[spec_win]['f'] > fmin,
                               spectra[spec_win]['f'] < fmax)).all(axis=0)
            break

    power_signal = np.trapz(p_signal[f_bool], dx=df_signal)
    return power_signal / power_noise, spec_win


def calc_stalta(event: Event,
                fmin: float, fmax: float,
                len_sta=100, len_lta=1000) -> float:
    from obspy.signal.trigger import classic_sta_lta
    tr_stalta = event.waveforms_VBB.select(channel='BHZ')[0].copy()
    tr_stalta.differentiate()
    tr_stalta.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=6,
                     zerophase=True)
    nsta = len_sta * tr_stalta.stats.sampling_rate
    nlta = len_lta * tr_stalta.stats.sampling_rate
    chf = classic_sta_lta(tr_stalta, nlta=nlta, nsta=nsta)
    tr_stalta.data = chf

    #plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
    #         tr_stalta.data)
    tr_stalta.trim(starttime=event.starttime,
                   endtime=event.endtime)
    #plt.plot(tr_stalta.times() + float(tr_stalta.stats.starttime),
    #         tr_stalta.data, 'r')
    #plt.savefig('./tmp/stalta_%s.png' % event.name)
    #plt.close('all')

    return np.max(tr_stalta.data)
