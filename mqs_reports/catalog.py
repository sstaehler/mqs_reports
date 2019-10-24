#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

from os.path import join as pjoin
from typing import Union

import matplotlib.ticker
import numpy as np
from mars_tools.insight_time import solify
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpldatacursor import datacursor
from obspy import UTCDateTime as utct
from tqdm import tqdm

from mqs_reports.annotations import Annotations
from mqs_reports.event import Event, EVENT_TYPES
from mqs_reports.magnitudes import M2_4, lorenz_att
from mqs_reports.scatter_annot import scatter_annot
from mqs_reports.utils import plot_spectrum, envelope_smooth, pred_spec


class Catalog:
    def __init__(self,
                 events=None,
                 fnam_quakeml='catalog.xml',
                 quality=('A', 'B', 'C'),
                 type_select='all'):
        """
        Class to hold catalog of multiple events. Initialized from
        dictionary with Events or QuakeML with Mars extensions.
        :param events: dictionary of events. If not set, the events are read
                       from QuakeML file
        :param event_tmp_dir: temporary directory for waveform files
        :param fnam_quakeml: Path to QuakeML file
        :param quality: Desired event quality
        :param type_select: Desired event types. Either direct type or
                            "all" for BB, HF and LF
                            "higher" for HF and BB
                            "lower" for LF and BB
        """
        from mqs_reports.read_BED_Mars import read_QuakeML_BED
        self.events = []
        if events is None:
            if type_select == 'all':
                type_des = EVENT_TYPES
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
            if quality == 'all':
                quality = ('A', 'B', 'C', 'D')
            self.types = type_des
            self.events.extend(read_QuakeML_BED(fnam=fnam_quakeml,
                                                event_type=type_des,
                                                quality=quality,
                                                phase_list=['start', 'end',
                                                            'P', 'S',
                                                            'Pg', 'Sg',
                                                            'Peak_M2.4',
                                                            'Peak_MbP',
                                                            'Peak_MbS',
                                                            'noise_start',
                                                            'noise_end',
                                                            'P_spectral_start',
                                                            'P_spectral_end',
                                                            'S_spectral_start',
                                                            'S_spectral_end']))
        else:
            if isinstance(events, Event):
                events = [events]
            if events:
                self.events.extend(events)
        pass

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return list(self.events).__iter__()

    def __add__(self, other):
        if isinstance(other, Event):
            other = Catalog([other])
        if not isinstance(other, Catalog):
            raise TypeError
        events = self.events + other.events
        return self.__class__(events=events)

    def __str__(self, extended=False):
        out = str(len(self.events)) + ' Events(s) in Catalog:\n'

        for event_type in EVENT_TYPES:
            n = len([e for e in self if e.mars_event_type == event_type])
            out += f'\n    {n:4d} {event_type} events:\n        '

            for Q in 'ABCD':
                nQ = len([e for e in self
                          if (e.mars_event_type == event_type and
                              e.quality == Q)])
                out += f'{nQ:4d} {Q} '

        out += '\n\n'

        if len(self.events) <= 20 or extended is True:
            out = out + "\n".join([str(_i) for _i in self])
        else:
            out = out + "\n" + self.events[0].__str__() + "\n" + \
                  '...\n(%i other events)\n...\n' % (len(self.events) - 2) + \
                  self.events[-1].__str__() + '\n\n[Use "print(' + \
                  'Catalog.__str__(extended=True))" to print all Events]'
        out += '\n'
        return out

    def select(self,
               name: Union[tuple, list, str] = None,
               event_type: Union[tuple, list, str] = None,
               quality: Union[tuple, list, str] = None
               ):
        """
        Return new Catalog object only with the events that match the given
        criteria (e.g. all with name=="S026?a").
        Criteria can either be given as string with wildcards or as tuple of
        allowed values.
        :param name: Name of the event ("SXXXXy")
        :param event_type: two-letter acronym "BB", "LF", "HF", "24", "VF
        :param quality: A to D
        :return:
        """
        from fnmatch import fnmatch
        events = []
        for event in self:
            # skip event if any given criterion is not matched
            if name is not None:
                if type(name) in (tuple, list):
                    if event.name not in name:
                        continue
                else:
                    if not fnmatch(event.name, name):
                        continue

            if event_type is not None:
                if type(event_type) in (tuple, list):
                    if (event.mars_event_type_short not in event_type and
                         event.mars_event_type not in event_type):
                        continue
                else:
                    if (not fnmatch(event.mars_event_type_short, event_type)
                         and not fnmatch(event.mars_event_type, event_type)):
                        continue

            if quality is not None:
                if type(quality) in (tuple, list):
                    if event.quality not in quality:
                        continue
                else:
                    if not fnmatch(event.quality, quality):
                        continue
            events.append(event)
        return self.__class__(events=events)

    def load_distances(self, fnam_csv):
        for event in self:
            event.load_distance_manual(fnam_csv)

    def calc_spectra(self, winlen_sec: float) -> None:
        """
        Add spectra to each Event object in Catalog.
        Spectra are stored in dictionaries
            event.spectra for VBB
            event.spectra_SP for SP
        Spectra are calculated separately for time windows "noise", "all",
        "P" and "S". If any of the necessary picks is missing, this entry is
        set to None.
        :param winlen_sec: window length for Welch estimator
        """
        for event in tqdm(self):
            event.calc_spectra(winlen_sec=winlen_sec)

    def read_waveforms(self,
                       inv,
                       sc3dir: str,
                       event_tmp_dir='./events',
                       kind: str = 'DISP') -> None:
        """
        Wrapper to check whether local copy of corrected waveform exists and
        read it from sc3dir otherwise (and create local copy)
        :param inv: Obspy.Inventory to use for instrument correction
        :param sc3dir: path to data, in SeisComp3 directory structure
        :param kind: 'DISP', 'VEL' or 'ACC'. Note that many other functions
                     expect the data to be in displacement
        """
        for event in tqdm(self):
            event.read_waveforms(inv=inv, kind=kind, sc3dir=sc3dir,
                                 event_tmp_dir=event_tmp_dir)

    def plot_pickdiffs(self, pick1_X, pick2_X, pick1_Y, pick2_Y, vX=None,
                       vY=None, fig=None, show=True, **kwargs):
        times_X = []
        times_Y = []
        names = []
        for event in self:
            try:
                # Remove events that do not have all four picks
                for pick in [pick1_X, pick1_Y, pick2_X, pick2_Y]:
                    assert not event.picks[pick] == ''
            except AssertionError:
                print('One or more picks missing for event %s' % event.name)
            else:
                times_X.append(utct(event.picks[pick1_X]) -
                               utct(event.picks[pick2_X]))
                times_Y.append(utct(event.picks[pick1_Y]) -
                               utct(event.picks[pick2_Y]))
                names.append(event.name)

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
        for event in self:
            try:
                # Remove events that do not have all four picks
                for pick in [pick1_Y, pick2_Y, 'start']:
                    assert not event.picks[pick] == ''
            except AssertionError:
                print('One or more picks missing for event %s' % event.name)
            else:
                times_X.append(float(solify(utct(event.picks['start']))) /
                               86400.)
                times_Y.append(utct(event.picks[pick1_Y]) -
                               utct(event.picks[pick2_Y]))
                names.append(event.name)

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

    def plot_24_alignment(
         self, pre_time=120., post_time=120., fmax_filt=2.7, fmin_filt=2.1,
         envelope_window=100., amp_fac=2., show_picks=True,
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2'},
         linestyle={'A': '-', 'B': '-', 'C': '--', 'D': ':'}, show=True):

        events = []
        for event in self:
            # filter for HF and 2.4 events
            if event.mars_event_type not in ['2.4_HZ', 'HIGH_FREQUENCY']:
                continue

            # Remove events that do not have all picks
            try:
                for pick in ['Pg', 'Sg', 'end', 'noise_start', 'noise_end']:
                    assert not event.picks[pick] == ''
            except:
                print('One or more picks missing for event %s' % (event.name))
            else:
                events.append(event)

        # compute TP - TS to sort by distance
        tt_PgSg = np.array([(utct(event.picks['Sg']) -
                             utct(event.picks['Pg'])) for event in events])
        sorted_ids = np.argsort(tt_PgSg)

        fig = plt.figure()

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
            trZ_env = envelope_smooth(envelope_window, trZ)
            trZ_noise_env = envelope_smooth(envelope_window, trZ_noise)

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
                     ls=linestyle[event.quality], zorder=1000-k)

            # fill between noise amplitude estimate and envelope
            plt.fill_between(X, Y0, Y,  where=((Y>=Y0) * (X>0) * (X<duration)),
                             color=colors[event.mars_event_type], alpha=0.4,
                             zorder=-20)

            if show_picks:
                plt.plot([tt, tt], [k, k+0.3*amp_fac], color='C8')
                plt.plot([duration, duration], [k, k+0.8], color='C9')

            # plot noise
            X = trZ_noise_env.times()
            X = X - pre_time - 400 - X[-1]
            Y = trZ_noise_env.data * amp_fac + k
            X = X[::10]
            Y = Y[::10]
            plt.plot(X, Y, color='k')

            plt.text(-pre_time, k + 0.5, event.name + ' ',
                     ha='right', va='center')

        # time 0 line
        plt.axvline(0, color='C4')

        # legend
        llabels = ['HIGH_FREQUENCY', '2.4_HZ']
        lcolors = [colors[l] for l in llabels]
        llines = [Line2D([0], [0], color=c) for c in lcolors]
        plt.legend(llines, llabels)

        # lable, limit, ticks
        plt.xlabel('time after Pg / s')
        plt.xlim(-pre_time - 300, None)
        plt.yticks([], [])

        if show:
            plt.show()
        else:
            return fig

    def plot_HF_spectra(self, SNR=2., tooltip=False, show=True):

        fig = plt.figure()

        cat = self.select(quality='B', event_type=['2.4_HZ', 'HIGH_FREQUENCY'])

        class ContinueI(Exception):
                pass

        for event in cat:

            # Skip events that do not have all picks, but print message in case
            try:
                for stype in ['P', 'S', 'noise']:
                    if not stype in event.spectra:
                        raise ContinueI(f'Missing spectral {stype} picks in event {event.name}')

                    if len(event.spectra[stype]) == 0:
                        raise ContinueI(f'Spectrum empty for {stype} in event {event.name}')
            except ContinueI as e:
                print(e)
                continue

            lw = 1.

            mask_P_1Hz = (event.spectra['P']['f'] > 0.86) * (event.spectra['P']['f'] < 1.14)

            mask_P = event.spectra['P']['f'] < 1.3
            mask_P += event.spectra['P']['f'] > 7.
            peak = event.spectra['P']['p_Z'][np.logical_not(mask_P)].max()
            mask_P = event.spectra['P']['f'] < 0.7
            mask_P += event.spectra['P']['f'] > 7.
            mask_P += mask_P_1Hz
            mask_P += event.spectra['P']['p_Z'] < SNR * event.spectra['noise']['p_Z']
            msP = np.ma.masked_where(mask_P, event.spectra['P']['p_Z'])
            msPN = np.ma.masked_where(mask_P_1Hz, event.spectra['P']['p_Z'])

            msP /= peak
            msPN /= peak

            l1, = plt.plot(event.spectra['P']['f'], 10 * np.log10(msP),
                           color='C0', alpha=1., label=f'{event.name}, P',
                           lw=lw)
            plt.plot(event.spectra['P']['f'], 10 * np.log10(msPN),
                     color='lightgray', zorder=-10, lw=lw,
                     label=f'{event.name}, P noise')

            mask_S_1Hz = (event.spectra['S']['f'] > 0.86) * (event.spectra['S']['f'] < 1.14)

            mask_S = event.spectra['S']['f'] < 1.3
            mask_S += event.spectra['S']['f'] > 7.
            peak = event.spectra['S']['p_Z'][np.logical_not(mask_S)].max()
            mask_S = event.spectra['S']['f'] < 0.7
            mask_S += event.spectra['S']['f'] > 7.
            mask_S += mask_S_1Hz
            mask_S += event.spectra['S']['p_Z'] < SNR * event.spectra['noise']['p_Z']
            msS = np.ma.masked_where(mask_S, event.spectra['S']['p_Z'])
            msSN = np.ma.masked_where(mask_S_1Hz, event.spectra['S']['p_Z'])

            msS /= peak
            msSN /= peak

            l2, = plt.plot(event.spectra['S']['f'], 10 * np.log10(msS),
                           color='C1', alpha=1., label=f'{event.name}, S',
                           lw=lw)
            plt.plot(event.spectra['S']['f'], 10 * np.log10(msSN),
                     color='lightgray', zorder=-10, lw=lw, label=f'{event.name}, S noise')

        if tooltip:
            datacursor(formatter='{label}'.format)

        # plot lorenz with attenuation
        f = np.linspace(0.01, 10., 1000)
        spec1 = lorenz_att(f, A0=-7, x0=2.4, tstar=0.1, xw=0.3, ampfac=15.)
        spec2 = lorenz_att(f, A0=-5, x0=2.4, tstar=0.2, xw=0.3, ampfac=15.)
        spec3 = lorenz_att(f, A0=-8, x0=2.4, tstar=0.05, xw=0.3, ampfac=15.)
        spec4 = lorenz_att(f, A0=-3, x0=2.4, tstar=0.3, xw=0.3, ampfac=15.)
        l3, = plt.plot(f, spec1, color='k', label='t* = 0.1')
        l4, = plt.plot(f, spec2, color='k', ls='--', label='t* = 0.2')
        l5, = plt.plot(f, spec3, color='k', ls='-.', label='t* = 0.05')
        l6, = plt.plot(f, spec4, color='k', ls=':', label='t* = 0.3')

        llabels = ['P', 'S'] + [l.get_label() for l in [l3, l4, l5, l6]]
        plt.legend([l1, l2, l3, l4, l5, l6], llabels)
        plt.xlabel('frequency / Hz')
        plt.ylabel('PSD relative to 2.4 peak amplitude')

        ax = plt.gca()
        ax.set_xscale('log')
        xmajor_locator = matplotlib.ticker.LogLocator(
            base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numdecs=4, numticks=None)
        ax.get_xaxis().set_major_locator(xmajor_locator)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.xlim(0.3, 8.)
        plt.ylim(-30., 7.)

        plt.title(f'Spectra for {len(cat)} events')

        if show:
            plt.show()
        else:
            return fig

    def plot_magnitude_distance(
         self, mag_type='m2.4',
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2'},
         markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
         markerfill={'A': True, 'B': True, 'C': False, 'D': False},
         show=True):

        fig = plt.figure()

        legend_elements = []

        for event_type in ['2.4_HZ', 'HIGH_FREQUENCY']:
            for quality in 'ABCD':
                cat = self.select(quality=quality, event_type=event_type)

                if len(cat) == 0:
                    continue

                # collect properties for plotting
                M, dist = np.array([
                    (event.magnitude(mag_type=mag_type, distance=event.distance),
                     event.distance) for event in cat]).T.astype(float)

                S = np.array([markersize[event.quality] for event in cat])
                names = np.array([f'{event.name} {event.duration_s:.0f}' for event in cat])

                mask = np.logical_not(np.isnan(M))
                M = M[mask]
                dist = dist[mask]
                S = S[mask]
                names = names[mask]

                if markerfill[quality]:
                    colorargs = {'c': colors[event_type]}
                else:
                    colorargs = {'edgecolors': colors[event_type],
                                 'facecolor': 'none'}

                scatter_annot(dist, M, s=S, fig=fig, names=names,
                              label=f'{event_type}, {quality}',
                              **colorargs)

        dist = np.linspace(3, 40)
        magc_24 = M2_4(-219, dist)
        magc_HF = M2_4(-212.5, dist)
        plt.plot(dist, magc_24, label='M2.4(-219.0 dB)', color='C3')
        plt.plot(dist, magc_HF, label='M2.4(-212.5 dB)', color='C3', ls='--')

        plt.xlabel('distance / degree')
        plt.ylabel('M2.4')
        plt.legend()

        if show:
            plt.show()
        else:
            return fig

    def plot_distance_hist(self, show=True):

        fig = plt.figure()

        bins = np.linspace(5 ** 2, 40 ** 2, 10) ** 0.5
        dists_all = [event.distance for event in self]
        dists_ABC = [event.distance for event in
                     self.select(quality=('A', 'B', 'C'))]
        dists_AB = [event.distance for event in
                     self.select(quality=('A', 'B'))]

        hist_all = np.histogram(dists_all, bins=bins)[0]
        hist_ABC = np.histogram(dists_ABC, bins=bins)[0]
        hist_AB = np.histogram(dists_AB, bins=bins)[0]

        for b1, b2, h1, h2, h3 in zip(bins[:-1], bins[1:], hist_all, hist_ABC,
                                      hist_AB):
            plt.plot([b1, b2], [h3, h3], color='C0', lw=1., marker='|')
            plt.plot([b1, b2], [h2, h2], color='C1', lw=1., marker='|')
            plt.plot([b1, b2], [h1, h1], color='C2', lw=1., marker='|')
            l3, = plt.plot([(b1+b2)/2], [h3], color='C0', marker='o', ms=8)
            l2, = plt.plot([(b1+b2)/2], [h2], color='C1', marker='o', ms=8)
            l1, = plt.plot([(b1+b2)/2], [h1], color='C2', marker='o', ms=8)

        plt.legend([l3, l2, l1], ['Quality B', 'Quality BC', 'Quality BCD'])
        plt.xlabel('distance / degree')
        plt.ylabel('# events per area')

        if show:
            plt.show()
        else:
            return fig

    def make_report(self,
                    dir_out: str = 'reports',
                    annotations: Annotations = None):
        """
        Create Magnitude report figure
        :param dir_out: Directory to write report to
        :param annotations: Annotations object; used to mark glitches,
                            if available
        """
        from os.path import exists as pexists
        for event in tqdm(self):
            fnam_report = pjoin(dir_out,
                                'mag_report_%s.html' %
                                event.name)
            if not pexists(fnam_report):
                event.make_report(fnam_out=fnam_report,
                                  annotations=annotations)
            else:
                event.fnam_report = fnam_report

    def plot_spectra(self,
                     ymin: float = -240.,
                     ymax: float = -170.,
                     fits: dict = None,
                     df_mute: object = 1.07) -> None:
        """
        Create big 6xnevent overview plot of all spectra.
        :param ymin: minimum y in dB
        :param ymax: maximum y in dB
        :param df_mute: percentage to mute around 1 Hz
        """
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
                                        plt.cm.tab20(
                                            np.linspace(0, 1, nevents))))
        ievent = 0
        for event in self:
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

                if fits is not None:
                    f = np.geomspace(0.01, 20., 100)
                    Mw = event.magnitude(mag_type='MFB')
                    p_pred = pred_spec(freqs=f,
                                       ds=1e6,
                                       mag=Mw,
                                       amp=fits[event.name]['A0'],
                                       Qm=fits[event.name]['Qm'],
                                       dist=event.distance * 55e3)
                    ax[iax, ichan].plot(f, p_pred)


            ax[iax, ichan + 2].legend()
            ax[iax, ichan].text(x=0.55, y=0.75, s=event.name,
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

    def write_table(self,
                    fnam_out: str = 'overview.html') -> None:
        """
        Create HTML overview table for catalog
        :param fnam_out: filename to write to
        """
        from mqs_reports.create_table import write_html

        write_html(self, fnam_out=fnam_out)
