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
from mqs_reports.utils import create_fnam_event, read_catalog, read_data, \
    __dayplot_set_x_ticks

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
import matplotlib.mlab as mlab
from obspy import UTCDateTime as utct
from obspy.signal.util import next_pow_2
from os.path import join as pjoin
from os import makedirs
from glob import glob
from argparse import ArgumentParser


def read_data_local(event_name, event):
    event_path = pjoin('events', '%s' % event_name)
    waveform_path = pjoin(event_path, 'waveforms')
    origin_path = pjoin(event_path, 'origin_id.txt')
    success = False
    if len(glob(origin_path)) > 0:
        with open(origin_path, 'r') as f:
            origin_local = f.readline().strip()
        if origin_local == event['origin_publicid']:
            event['waveforms_VBB'] = obspy.read(pjoin(waveform_path,
                                                      'waveforms_VBB.mseed'))
            SP_path = pjoin(waveform_path, 'waveforms_SP.mseed')
            if len(glob(SP_path)):
                event['waveforms_SP'] = obspy.read(SP_path)
            else:
                event['waveforms_SP'] = None
            success = True
    return success

def write_data_local(event_name, event):
    event_path = pjoin('events', '%s' % event_name)
    waveform_path = pjoin(event_path, 'waveforms')
    origin_path = pjoin(event_path, 'origin_id.txt')
    makedirs(waveform_path, exist_ok=True)

    with open(origin_path, 'w') as f:
        f.write(event['origin_publicid'])
    event['waveforms_VBB'].write(pjoin(waveform_path,
                                   'waveforms_VBB.mseed'),
                                 format='MSEED', encoding='FLOAT64')
    if event['waveforms_SP'] is not None and len(event['waveforms_SP']) > 0:
        event['waveforms_SP'].write(pjoin(waveform_path,
                                          'waveforms_SP.mseed'),
                                    format='MSEED', encoding='FLOAT64')


def calc_spectra_events(event_list, inv, kind, sc3dir, winlen_sec,
                        filenam_VBB_HG='XB.ELYSE.02.?H?.D.2019.%03d',
                        filenam_SP_HG='XB.ELYSE.65.EH?.D.2019.%03d'):
    events = dict()
    for cat_name, event_cat in event_list.items():
        for event_name, event in event_cat.items():
            print(event_name)
            if not read_data_local(event_name, event):
                read_data_from_sc3dir(event, filenam_SP_HG, filenam_VBB_HG, inv,
                                      kind, sc3dir)
                write_data_local(event_name, event)

            fnam_spectrum = pjoin('spectrum', 'spectrum_' + event_name)
            twins = (((event['picks']['start']),
                      (event['picks']['end'])),
                     ((event['picks']['noise_start']),
                      (event['picks']['noise_end'])),
                     ((event['picks']['P_spectral_start']),
                      (event['picks']['P_spectral_end'])),
                     ((event['picks']['S_spectral_start']),
                      (event['picks']['S_spectral_end'])))
            event['spectra'] = dict()
            event['spectra_SP'] = dict()
            variables = ('all',
                         'noise',
                         'P',
                         'S')
            for twin, variable in zip(twins, variables):
                event['spectra'][variable] = dict()
                if len(twin[0]) == 0:
                    continue
                for chan in ['Z', 'N', 'E']:
                    # f, p = read_spectrum(fnam_base=fnam_spectrum,
                    #                      variable=variable,
                    #                      chan=chan,
                    #                      origin_publicid=event[
                    #                          'origin_publicid'])
                    # if f is None:
                    st_sel = event['waveforms_VBB'].select(
                        channel='??' + chan)
                    if len(st_sel) > 0:
                        tr = st_sel[0].slice(starttime=utct(twin[0]),
                                             endtime=utct(twin[1]))
                        if tr.stats.npts > 0:
                            f, p = calc_PSD(tr,
                                            winlen_sec=winlen_sec)
                            event['spectra'][variable]['p_' + chan] = p
                        else:
                            f = np.arange(0, 1, 0.1)
                            p = np.zeros((10))
                    # else:
                    #     event['spectra'][variable]['p_' + chan] = p
                    event['spectra'][variable]['f'] = f

                if event['waveforms_SP'] is not None:
                    event['spectra_SP'][variable] = dict()
                    for chan in ['Z', 'N', 'E']:
                        st_sel = event['waveforms_SP'].select(
                            channel='??' + chan)
                        if len(st_sel) > 0:
                            tr = st_sel[0].slice(starttime=utct(twin[0]),
                                                 endtime=utct(twin[1]))
                            if tr.stats.npts > 0:
                                f, p = calc_PSD(tr,
                                                winlen_sec=winlen_sec)
                                event['spectra_SP'][variable]['p_' + chan] = p
                            else:
                                f = np.arange(0, 1, 0.1)
                                p = np.zeros((10))
                                event['spectra_SP'][variable]['p_' + chan] = p
                                event['spectra_SP'][variable]['f_' + chan] = f
                        else:
                            # Case that only SP1==SPZ is switched on
                            event['spectra_SP'][variable]['p_' + chan] = \
                                np.zeros_like(p)
                    event['spectra_SP'][variable]['f'] = f

            events[event_name] = event
            #write_spectrum(fnam_spectrum, event['spectra'],
            # event['origin_publicid'])
    return events


def read_data_from_sc3dir(event, filenam_SP_HG, filenam_VBB_HG, inv, kind,
                          sc3dir):
    fnam_VBB, fnam_SP = create_fnam_event(
        filenam_VBB_HG=filenam_VBB_HG,
        filenam_SP_HG=filenam_SP_HG,
        sc3dir=sc3dir, time=event['picks']['start'])
    event['picks'] = event['picks']
    if len(event['picks']['noise_start']) > 0:
        twin_start = min((utct(event['picks']['start']),
                          utct(event['picks']['noise_start'])))
    else:
        twin_start = utct(event['picks']['start'])
    if len(event['picks']['noise_end']) > 0:
        twin_end = max((utct(event['picks']['end']),
                        utct(event['picks']['noise_end'])))
    else:
        twin_end = utct(event['picks']['end'])
    if len(glob(fnam_SP)) > 0:
        # Use SP waveforms only if 65.EH? exists, not otherwise (we
        # don't need 20sps SP data)
        event['waveforms_SP'] = read_data(fnam_SP, inv=inv, kind=kind,
                                          twin=[twin_start - 100.,
                                                twin_end + 100.],
                                          fmin=0.5)
    else:
        event['waveforms_SP'] = None
    event['waveforms_VBB'] = read_data(fnam_VBB, inv=inv,
                                       kind=kind,
                                       twin=[twin_start - 900.,
                                             twin_end + 900.])


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


def plot_events_new(events, ymin, ymax, df_mute=1.07):
    nevents = len(events)
    nrows = max(2, (nevents + 1) // 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=6, figsize=(14, 10),
                           sharex='all', sharey='all')
    fig_all, ax_all = plt.subplots(nrows=1, ncols=2,
                                   sharex='all', sharey='all',
                                   figsize=(12, 6))
    iax = 0
    second = 0
    for a in ax_all:
        a.set_prop_cycle(plt.cycler('color',
                                    plt.cm.tab20(np.linspace(0, 1, nevents))))
    ievent = 0
    for event_name, event in events.items():
        print(event_name)
        ichan = 0
        if iax == nrows:
            iax -= nrows
            ichan = 3
            second = 1
        if second == 1:
            ichan = 3

        bodywave = False
        spectrum = event['spectra']['noise']
        if len(spectrum) > 0:
            plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                          spectrum, fmax=8., color='k', label='noise')
        spectrum = event['spectra']['P']
        if len(spectrum) > 0:
            plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                          spectrum, fmax=8., color='b', label='P-coda')
            bodywave = True
        spectrum = event['spectra']['S']
        if len(spectrum) > 0:
            plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                          spectrum, fmax=8., color='g', label='S-code')
            bodywave = True
        spectrum = event['spectra']['all']
        if len(spectrum) > 0 and not bodywave:
            plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                          spectrum, fmax=8., color='r', label='total')

        if len(event['spectra_SP']) > 0:
            if 'noise' in event['spectra_SP']:
                spectrum = event['spectra_SP']['noise']
                if len(spectrum) > 0:
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                  fmin=7., color='k')  # , label='noise')
            if 'P' in event['spectra_SP']:
                spectrum = event['spectra_SP']['P']
                if len(spectrum) > 0:
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                  fmin=7., color='b')  # , label='P-coda')
                    bodywave = True
            if 'S' in event['spectra_SP']:
                spectrum = event['spectra_SP']['S']
                if len(spectrum) > 0:
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                  fmin=7., color='g')  # , label='S-code')
                    bodywave = True
            spectrum = event['spectra_SP']['all']
            if len(spectrum) > 0 and not bodywave:
                plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                              fmin=7., color='r')  # , label='total')

        ax[iax, ichan + 2].legend()
        ax[iax, ichan].text(x=0.55, y=0.75, s=event_name,
                            fontsize=14,
                            transform=ax[iax, ichan].transAxes)

        iax += 1
        ievent += 1

    ax_all[0].set_xscale('log')
    ax_all[0].set_xlim(0.03, 5)
    ax_all[0].set_title('vertical', fontsize=18)
    ax_all[1].set_title('sum(horizontals)', fontsize=18)
    ax_all[0].set_xlabel('frequency [Hz]', fontsize=16)
    ax_all[1].set_xlabel('frequency [Hz]', fontsize=16)
    ax_all[0].set_yticks((-140, -160, -180, -200, -220, -240))
    ax_all[0].set_yticklabels((-140, -160, -180, -200, -220, -240))
    ax_all[0].set_ylim(ymin, ymax)
    ax_all[0].legend()
    ax[0][0].set_xscale('log')
    ax[0][0].set_yticks((-140, -160, -180, -200, -220, -240))
    ax[0][0].set_yticklabels((-140, -160, -180, -200, -220, -240))
    ax[0][0].set_xlim(0.1, 20)
    ax[0][0].set_ylim(ymin, ymax)
    ax[0][0].set_title('vertical', fontsize=18)
    ax[0][1].set_title('north/south', fontsize=18)
    ax[0][2].set_title('east/west', fontsize=18)
    ax[0][3].set_title('vertical', fontsize=18)
    ax[0][4].set_title('north/south', fontsize=18)
    ax[0][5].set_title('east/west', fontsize=18)
    fig.subplots_adjust(bottom=0.05, top=0.95, wspace=0.05, hspace=0.05,
                        left=0.1, right=0.98)

    string = 'displacement PSD [m$^2$]/Hz'
    ax[(nevents + 1) // 4][0].set_ylabel(string, fontsize=13)
    for a in [ax[-1][1], ax[-1][4]]:
        a.set_xlabel('frequency / Hz', fontsize=12)
    ax[-1][-1].legend()
    plt.tight_layout()

    plt.show()


def plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                  fmin=0.1, fmax=100.,
                  **kwargs):
    f = spectrum['f']
    for chan in ['Z', 'N', 'E']:
        try:
            p = spectrum['p_' + chan]
        except(KeyError):
            continue
        else:
            bol_1Hz_mask = np.array(
                (np.array((f > fmin, f < fmax)).all(axis=0),
                 np.array((f < 1. / df_mute,
                           f > df_mute)).any(axis=0))
                ).all(axis=0)

            ax[iax, ichan].plot(f[bol_1Hz_mask],
                                10 * np.log10(p[bol_1Hz_mask]),
                                **kwargs)
            bol_1Hz_mask = np.invert(bol_1Hz_mask)
            p = np.ma.masked_where(condition=bol_1Hz_mask, a=p,
                                   copy=False)

            if ichan % 3 == 0:
                ax_all[ichan % 3].plot(f,
                                       10 * np.log10(p),
                                       lw=0.5, c='lightgrey', zorder=1)
            elif ichan % 3 == 1:
                tmp2 = p
            elif ichan % 3 == 2:
                ax_all[ichan % 3 - 1].plot(f,
                                           10 * np.log10(tmp2 + p),
                                           lw=0.5, c='lightgrey', zorder=1)
            ichan += 1


def calc_PSD(tr, winlen_sec):
    Fs = tr.stats.sampling_rate
    winlen = min(winlen_sec * Fs,
                 (tr.stats.endtime -
                  tr.stats.starttime) * Fs / 4.)
    NFFT = next_pow_2(winlen)
    p, f = mlab.psd(tr.data,
                    Fs=Fs, NFFT=NFFT, detrend='linear',
                    pad_to=NFFT * 2, noverlap=NFFT // 2)
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


# def download_data_and_save_local(events):
#     events = dict()
#     for cat_name, event_cat in event_list.items():
#         for event_name, event in event_cat.items():
#             print(event_name)
#             if not read_data_local(event_name, event):
#                 read_data_from_sc3dir(event, filenam_SP_HG, filenam_VBB_HG, inv,
#                                       kind, sc3dir)
#                 write_data_local(event_name, event)

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

    event_list = read_catalog(fnam_quakeml=args.input_quakeml,
                              type_select='higher',
                              quality=['A', 'B', 'C'])
    events = calc_spectra_events(event_list, inv, kind, args.sc3_dir,
                                 winlen_sec)
    plot_events_new(events, ymin=-240, ymax=-170)
