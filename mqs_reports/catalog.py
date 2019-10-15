#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from obspy.signal.filter import envelope
from os.path import join as pjoin

import numpy as np
from mars_tools.insight_time import solify
from matplotlib import pyplot as plt
from obspy import UTCDateTime as utct
from tqdm import tqdm

from mqs_reports.scatter_annot import scatter_annot


class Catalog:
    def __init__(self,
                 fnam_quakeml='catalog.xml',
                 quality=('A', 'B', 'C'),
                 type_select='all'):
        """
        Class to hold catalog of multiple events. Initialized from QuakeML
        with Mars extensions.
        :param fnam_quakeml: Path to QuakeML file
        :param quality: Desired event quality
        :param type_select: Desired event types. Either direct type or
                            "all" for BB, HF and LF
                            "higher" for HF and BB
                            "lower" for LF and BB
        """
        from mqs_reports.read_BED_Mars import read_QuakeML_BED

        if type_select == 'all':
            type_des = ['BROADBAND',
                        'HIGH_FREQUENCY',
                        'LOW_FREQUENCY',
                        '2.4_HZ']
        elif type_select == 'higher':
            type_des = ['HIGH_FREQUENCY',
                        'BROADBAND']
        elif type_select == 'lower':
            type_des = ['LOW_FREQUENCY',
                        'BROADBAND']
        else:
            if len(type_select) == 1:
                type_des = [type_select]
            else:
                type_des = type_select
        self.types = type_des
        self.events = read_QuakeML_BED(fnam=fnam_quakeml,
                                       event_type=type_des,
                                       quality=quality,
                                       phase_list=['start', 'end',
                                                   'P', 'S',
                                                   'Pg', 'Sg',
                                                   'Peak_M2.4',
                                                   'Peak_MbP',
                                                   'Peak_MbS',
                                                   'noise_start', 'noise_end',
                                                   'P_spectral_start',
                                                   'P_spectral_end',
                                                   'S_spectral_start',
                                                   'S_spectral_end'])


    def calc_spectra(self, winlen_sec):
        for event_name, event in tqdm(self.events.items()):
            event.calc_spectra(winlen_sec=winlen_sec)

    def read_waveforms(self, inv, kind, sc3dir):
        for event_name, event in tqdm(self.events.items()):
            event.read_waveforms(inv=inv, kind=kind, sc3dir=sc3dir)


    def plot_pickdiffs(self, pick1_X, pick2_X, pick1_Y, pick2_Y, vX=None,
                       vY=None, fig=None, show=True, **kwargs):
        times_X = []
        times_Y = []
        names = []
        for name, event in self.events.items():
            try:
                # Remove events that do not have all four picks
                for pick in [pick1_X, pick1_Y, pick2_X, pick2_Y]:
                    assert not event.picks[pick] == ''
            except:
                print('One or more picks missing for event %s' % (name))
            else:
                times_X.append(utct(event.picks[pick1_X]) -
                               utct(event.picks[pick2_X]))
                times_Y.append(utct(event.picks[pick1_Y]) -
                               utct(event.picks[pick2_Y]))
                names.append(name)

        if fig is None:
            fig = plt.figure()
        if vX is not None:
            times_X = np.asarray(times_X) * vX
        if vY is not None:
            times_Y = np.asarray(times_Y) * vY

        fig, ax = scatter_annot(times_X, times_Y, fig=fig,
                                names=names,
                                **kwargs)
        if vX is None:
            ax.set_xlabel('$T_{%s} - T_{%s}$' % (pick1_X, pick2_X))
        else:
            ax.set_xlabel('distance / km (from %s-%s)' % (pick1_X, pick2_X))
        if vY is None:
            ax.set_ylabel('$T_{%s} - T_{%s}$' % (pick1_Y, pick2_Y))
        else:
            ax.set_ylabel('distance / km (from %s-%s)' % (pick1_Y, pick2_Y))

        if show:
            plt.show()


    def plot_pickdiff_over_time(self, pick1_Y, pick2_Y, vY=None,
                                fig=None, show=True, **kwargs):
        times_X = []
        times_Y = []
        names = []
        for name, event in self.events.items():
            try:
                # Remove events that do not have all four picks
                for pick in [pick1_Y, pick2_Y, 'start']:
                    assert not event.picks[pick] == ''
            except:
                print('One or more picks missing for event %s' % (name))
            else:
                times_X.append(float(solify(utct(event.picks['start']))) /
                                     86400.)
                times_Y.append(utct(event.picks[pick1_Y]) -
                               utct(event.picks[pick2_Y]))
                names.append(name)

        if fig is None:
            fig = plt.figure()

        if vY is not None:
            times_Y = np.asarray(times_Y) * vY

        fig, ax = scatter_annot(times_X, times_Y, fig=fig,
                                names=names, **kwargs)

        ax.set_xlabel('Sol')
        if vY is None:
            ax.set_ylabel('$T_{%s} - T_{%s}$' % (pick1_Y, pick2_Y))
        else:
            ax.set_ylabel('distance / km (from %s-%s)' % (pick1_Y, pick2_Y))

        if show:
            plt.show()

    def plot_24_alignment(self, show=True, pre_time=120., post_time=120.,
                          fmax_filt=2.7, fmin_filt=2.1, envelope_window=100.,
                          amp_fac=2., show_picks=True,
                          colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2'},
                          linestyle={'A': '-', 'B': '-', 'C': '--', 'D': ':'}):
        events = []
        for name, event in self.events.items():
            try:
                # Remove events that do not have all picks
                for pick in ['Pg', 'Sg', 'end', 'noise_start', 'noise_end']:
                    assert not event.picks[pick] == ''
            except:
                print('One or more picks missing for event %s' % (name))
            else:
                events.append(event)

        # compute TP - TS to sort by distance
        tt_PgSg = np.array([(utct(event.picks['Sg']) -
                             utct(event.picks['Pg'])) for event in events])
        sorted_ids = np.argsort(tt_PgSg)

        def _envelope(tr):
            tr_env = tr.copy()
            tr_env.data = envelope(tr_env.data)

            w = np.ones(int(envelope_window / tr.stats.delta))
            w /= w.sum()
            tr_env.data = np.convolve(tr_env.data, w, 'valid')

            return tr_env

        for k, i in enumerate(sorted_ids):
            event = events[i]
            tt = tt_PgSg[i]
            duration = utct(event.picks['end']) - utct(event.picks['Pg'])

            # slice to time window
            trZ = event.waveforms_VBB.select(channel='??Z')[0].copy()
            starttime = utct(event.picks['Pg']) - pre_time - envelope_window
            endtime = utct(event.picks['end']) + post_time + envelope_window

            if (starttime - trZ.stats.starttime) < 0.:
                print(event.name, ': starttime problem')
                start_shift = - (starttime - trZ.stats.starttime)
            else:
                start_shift = 0.

            if (endtime - trZ.stats.endtime) > 0.:
                print(event.name, ': endtime problem')

            trZ = trZ.slice(starttime=starttime, endtime=endtime)

            # noise time window
            trZ_noise = event.waveforms_VBB.select(channel='??Z')[0].copy()
            starttime = utct(event.picks['noise_start'])
            endtime = utct(event.picks['noise_end'])
            trZ_noise = trZ_noise.slice(starttime=starttime, endtime=endtime)

            # detrend + filter
            for tr in [trZ, trZ_noise]:
                tr.detrend()
                tr.filter('lowpass', freq=fmax_filt, corners=8)
                tr.filter('highpass', freq=fmin_filt, corners=8)

            # compute envelopes
            trZ_env = _envelope(trZ)
            trZ_noise_env = _envelope(trZ_noise)

            # get max during the event for scaling
            trZ_env_event = trZ_env.slice(starttime=utct(event.picks['Pg']),
                                          endtime=utct(event.picks['end']))

            scaling = trZ_env_event.data.max()
            trZ_env.data /= scaling
            trZ_noise_env.data /= scaling

            # setup plotting variables
            X = trZ_env.times() - pre_time + start_shift
            #Y = trZ_env.data * amp_fac + k
            #Y0 = np.median(trZ_noise_env.data) * amp_fac + k
            Y = (trZ_env.data - np.median(trZ_noise_env.data)) * amp_fac + k
            Y0 = k

            # downsample to speed up plotting
            X = X[::10]
            Y = Y[::10]

            plt.plot(X, Y, color=colors[event.mars_event_type],
                     ls=linestyle[event.quality])

            # fill between noise amplitude estimate and envelope
            plt.fill_between(X, Y0, Y,  where=((Y>=Y0) * (X>0) * (X<duration)),
                             color='lightgray', alpha=1., zorder=-20)

            if show_picks:
                plt.plot([tt, tt], [k, k+0.8], color='C8')
                plt.plot([duration, duration], [k, k+0.8], color='C9')

            # plot noise
            X = trZ_noise_env.times()
            X = X - pre_time - 200 - X[-1]
            Y = trZ_noise_env.data * amp_fac + k
            X = X[::10]
            Y = Y[::10]
            plt.plot(X, Y, color='k')

            plt.text(-pre_time, k + 0.5, event.name + ' ',
                     ha='right', va='center')

        plt.axvline(0, color='C4')

        plt.xlabel('time after Pg / s')
        plt.xlim(-pre_time - 200, None)
        plt.yticks([], [])

        plt.show()


    def make_report(self, dir_out='reports', annotations=None):
        from os.path import exists as pexists
        for name, event in tqdm(self.events.items()):
            fnam_report = pjoin(dir_out,
                                'mag_report_%s.html' %
                                name)
            if not pexists(fnam_report):
                event.make_report(fnam_out=fnam_report,
                                  annotations=annotations)
            else:
                event.fnam_report = fnam_report

    def plot_spectra(self, event_list='all', ymin=-240, ymax=-170,
                     df_mute=1.07):
        nevents = len(self.events)
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
        for event_name, event in self.events.items():
            ichan = 0
            if iax == nrows:
                iax -= nrows
                ichan = 3
                second = 1
            if second == 1:
                ichan = 3

            bodywave = False
            spectrum = event.spectra['noise']
            if len(spectrum) > 0:
                plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                              spectrum, fmax=8., color='k', label='noise')
            spectrum = event.spectra['P']
            if len(spectrum) > 0:
                plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                              spectrum, fmax=8., color='b', label='P-coda')
                bodywave = True
            spectrum = event.spectra['S']
            if len(spectrum) > 0:
                plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                              spectrum, fmax=8., color='g', label='S-code')
                bodywave = True
            spectrum = event.spectra['all']
            if len(spectrum) > 0 and not bodywave:
                plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                              spectrum, fmax=8., color='r', label='total')

            if len(event.spectra_SP) > 0:
                if 'noise' in event.spectra_SP:
                    spectrum = event.spectra_SP['noise']
                    if len(spectrum) > 0:
                        plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                      fmin=7., color='k')  # , label='noise')
                if 'P' in event.spectra_SP:
                    spectrum = event.spectra_SP['P']
                    if len(spectrum) > 0:
                        plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                      fmin=7., color='b')  # , label='P-coda')
                        bodywave = True
                if 'S' in event.spectra_SP:
                    spectrum = event.spectra_SP['S']
                    if len(spectrum) > 0:
                        plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                                      fmin=7., color='g')  # , label='S-code')
                        bodywave = True
                spectrum = event.spectra_SP['all']
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

    def write_table(self, fnam_out='overview.html'):
        from mqs_reports.create_table import write_html

        write_html(self, fnam_out=fnam_out)

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
