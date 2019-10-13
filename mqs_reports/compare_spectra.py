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
from mqs_reports.catalog import Catalog
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
    args = define_arguments()
    # for type in ('all', 'HIGH_FREQUENCY', 'lower', 'HQ'):
    inv = obspy.read_inventory(args.input_inventory)
    kind = 'DISP'
    winlen_sec = 20.

    events = Catalog(fnam_quakeml=args.input_quakeml,
                     type_select='higher',
                     quality=['A', 'B', ])
    events.read_waveforms(inv, kind, args.sc3_dir)

    for i, event in events.events.items():
        print(  # '%s, %10.3e, %10.e' %
            i,
            event.pick_amplitude('Peak_MbP',
                                 comp='vertical',
                                 fmin=1. / 6.,
                                 fmax=1. / 2),
            event.pick_amplitude('Peak_MbS',
                                 comp='vertical',
                                 fmin=1. / 6.,
                                 fmax=1. / 2),
            event.pick_amplitude('Peak_M2.4',
                                 comp='vertical',
                                 fmin=2.2, fmax=2.6))
    events.calc_spectra(winlen_sec)
    events.plot_spectra(ymin=-240, ymax=-170)
