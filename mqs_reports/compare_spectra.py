#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to calculate and plot spectra for all MQS events so far

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2018
    Martin van Driel (Martin@vanDriel.de)
:license:
    None
"""
from mqs_reports.utils import __dayplot_set_x_ticks

__author__ = "Simon Stähler"

from os import environ

try:
    environ['DISPLAY']
except KeyError:
    import matplotlib

    matplotlib.use('Agg')

import numpy as np
import obspy
import matplotlib.pyplot as plt
from glob import glob
from argparse import ArgumentParser


def write_spectrum(fnam_base, spectrum, origin_publicid,
                   variables=('all', 'noise', 'P', 'S'),
                   chans=('Z', 'N', 'E')):
    for variable in variables:
        for chan in chans:
            if 'f' in spectrum[variable] and \
                    'p_' + chan in spectrum[variable]:
                fnam_out = fnam_base + '_' + chan + '_' + variable + '.txt'
                np.savetxt(fname=fnam_out,
                           header=origin_publicid,
                           X=np.asarray(
                               (spectrum[variable]['f'],
                                spectrum[variable]['p_' + chan])).T)


def read_spectrum(fnam_base, variable, chan, origin_publicid):
    fnam_in = fnam_base + '_' + chan + '_' + variable + '.txt'
    f = None
    p = None
    if len(glob(fnam_in)) > 0:
        with open(fnam_in, 'r') as f:
            origin_file = f.readline().strip()
        if origin_file[2:]==origin_publicid:
            dat = np.loadtxt(fnam_in, skiprows=1, usecols=(0, 1))
            f = dat[:, 0]
            p = dat[:, 1]
    return f, p


def plot_waveforms(st_event, st_noise, st_all, flims, fnam):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex='col',
                           sharey='col')

    for ichan, chan in enumerate(['Z', 'N', 'E']):
        for st_tmp, col, label in zip([st_all, st_noise, st_event],
                                      ['k', 'b', 'r'],
                                      ['all', 'noise', 'event']):
            st = st_tmp.copy()
            st.filter('highpass', freq=flims[0], zerophase=True)
            for tr in st:
                if flims[1] < tr.stats.sampling_rate / 2:
                    tr.filter('lowpass', freq=flims[1], zerophase=True)
            x = float(st[ichan].stats.starttime) + st[ichan].times()
            y = st[ichan].data
            ax[ichan].plot(x, y, c=col, label=label)
        ax[ichan].set_title('channel BH%s' % chan)
        ax[ichan].grid()
    ax[0].legend()
    __dayplot_set_x_ticks(ax[-1], starttime=st_all[0].stats.starttime,
                          endtime=st_all[0].stats.endtime)
    plt.tight_layout()
    fig.savefig(fnam, dpi=200)
    plt.close()


def define_arguments():
    helptext = 'Plot spectra of all events defined in CSV file'
    parser = ArgumentParser(description=helptext)

    helptext = "Input QuakeML BED file"
    parser.add_argument('input_quakeml', help=helptext,
                        default='catalog_20191002.xml')

    helptext = "Inventory file"
    parser.add_argument('input_inventory', help=helptext)

    helptext = "Path to SC3DIR"
    parser.add_argument('sc3_dir', help=helptext,
                        default='/mnt/mnt_sc3data')

    return parser.parse_args()


if __name__ == '__main__':
    from mqs_reports.catalog import Catalog

    args = define_arguments()
    # for type in ('all', 'HIGH_FREQUENCY', 'lower', 'HQ'):
    inv = obspy.read_inventory(args.input_inventory)
    kind = 'DISP'
    winlen_sec = 20.

    events = Catalog(fnam_quakeml=args.input_quakeml,
                     type_select='all',
                     quality=['A', 'B', 'C', 'D'])
    print(events.select(endtime=obspy.UTCDateTime('2019-09-30')))
    names = ['S0128a', 'S0218a', 'S0263a', 'S0264e', 'S0260a',
             'S0239a',
             'S0167a', 'S0226b', 'S0234c', 'S0185a', 'S0183a',
             'S0205a', 'S0235b', 'S0173a', 'S0154a', 'S0325a']
    events = events.select(name=names)
    # events = events.select(name=['S0105a',
    #                             'S0260a',
    #                             'S0290b',
    #                             'S0167a',
    #                             'S0133a',
    #                             'S0173a',
    #                             'S0128a',
    #                             'S0235b'])
    events.read_waveforms(inv=inv, sc3dir=args.sc3_dir,
                          )
    events.load_distances(fnam_csv='./mqs_reports/data/manual_distances.csv')
    fits = {'S0105a': {'Qm': 300, 'phase': 'S'},
            'S0260a': {'Qm': 2500, 'phase': 'P'},
            'S0263a': {'A0': -212, 'Qm': 2500, 'phase': 'P'},
            'S0264e': {'A0': -210, 'Qm': 2500, 'phase': 'P'},
            'S0239a': {'A0': -216, 'Qm': 2500, 'phase': 'P'},
            'S0290b': {'Qm': 2500, 'phase': 'P'},
            'S0167a': {'Qm': 1000, 'phase': 'S'},
            'S0133a': {'Qm': 1100, 'phase': 'S'},
            'S0234c': {'Qm': 1100, 'phase': 'S'},
            'S0226b': {'Qm': 300, 'phase': 'S'},
            'S0325a': {'Qm': 300, 'phase': 'S'},
            'S0205a': {'Qm': 300, 'phase': 'P'},
            'S0183a': {'Qm': 300, 'phase': 'P'},
            'S0154a': {'Qm': 1100, 'phase': 'S'},
            'S0185a': {'A0': -200, 'Qm': 1100, 'phase': 'S'},
            'S0173a': {'Qm': 320, 'phase': 'S'},
            'S0128a': {'Qm': 2500, 'phase': 'P'},
            'S0218a': {'Qm': 2500, 'phase': 'P'},
            'S0235b': {'Qm': 320, 'phase': 'S'}}

    # for event in events:
    # print(  # '%s, %10.3e, %10.e' %
    #     event.pick_amplitude('Peak_MbP',
    #                          comp='vertical',
    #                          fmin=1. / 6.,
    #                          fmax=1. / 2),
    #     event.pick_amplitude('Peak_MbS',
    #                          comp='vertical',
    #                          fmin=1. / 6.,
    #                          fmax=1. / 2),
    #     event.pick_amplitude('Peak_M2.4',
    #                          comp='vertical',
    #                          fmin=2.2, fmax=2.6))
    events.calc_spectra(winlen_sec)
    events.plot_spectra(ymin=-240, ymax=-170, fits=fits)
    events.make_report(dir_out='reports')
    events.write_table('table_SI5.1.html')
