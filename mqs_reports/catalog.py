#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

from os.path import join as pjoin, exists as pexists
from sys import stdout as stdout
from typing import Union

import matplotlib.pylab as pl
import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime as utct
from obspy.geodetics.base import degrees2kilometers
from scipy import stats
from tqdm import tqdm

from mqs_reports.annotations import Annotations
from mqs_reports.event import Event, EVENT_TYPES_PRINT, EVENT_TYPES_SHORT, \
    EVENT_TYPES, RADIUS_MARS, CRUST_VS, CRUST_VP
from mqs_reports.magnitudes import lorentz_att
from mqs_reports.scatter_annot import scatter_annot
from mqs_reports.snr import calc_stalta
from mqs_reports.utils import plot_spectrum, envelope_smooth, pred_spec, solify


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
            elif type_select == 'noSF':
                type_des = ['HIGH_FREQUENCY',
                            'VERY_HIGH_FREQUENCY',
                            'LOW_FREQUENCY',
                            '2.4_HZ',
                            'BROADBAND']
            elif type_select == 'higher':
                type_des = ['HIGH_FREQUENCY',
                            'BROADBAND']
            elif type_select == 'lower':
                type_des = ['LOW_FREQUENCY',
                            'BROADBAND']
            elif isinstance(type_select, str):
                type_des = [type_select]
            elif isinstance(type_select, list):
                type_des = type_select
            else:
                raise ValueError
            if quality == 'all':
                quality = ('A', 'B', 'C', 'D')
            self.types = type_des
            self.events.extend(read_QuakeML_BED(fnam=fnam_quakeml,
                                                event_type=type_des,
                                                quality=quality,
                                                phase_list=['start', 'end',
                                                            'P', 'S',
                                                            'PP', 'SS',
                                                            'Pg', 'Sg',
                                                            'Peak_M2.4',
                                                            'Peak_MbP',
                                                            'Peak_MbS',
                                                            'x1', 'x2', 'x3',
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

    def origin_vs_start_time(self):
        dt = [(ev.name, utct(ev.picks['start']) - ev.origin_time)
              for ev in self]

        dt = sorted(dt, key=lambda x: x[1], reverse=True)

        for item in dt:
            print(item[0], item[1])


    def select(self,
               name: Union[tuple, list, str] = None,
               event_type: Union[tuple, list, str] = None,
               quality: Union[tuple, list, str] = None,
               distmin: float = None,
               distmax: float = None,
               starttime: utct = None,
               endtime: utct = None,
               ):
        """
        Return new Catalog object only with the events that match the given
        criteria (e.g. all with name=="S026?a").
        Criteria can either be given as string with wildcards or as tuple of
        allowed values.
        :param name: Name of the event ("SXXXXy")
        :param event_type: two-letter acronym "BB", "LF", "HF", "24", "VF, "SF"
        :param quality: A to D
        :param distmin: minimum distance (in degree)
        :param distmax: maximum distance (in degree)
        :param starttime: minimum origin time (in UTC)
        :param endtime: maximum origin time (in UTC)
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

            if distmin is not None:
                if event.distance is None or event.distance < distmin:
                    continue

            if distmax is not None:
                if event.distance is None or event.distance > distmax:
                    continue

            if starttime is not None:
                if event.starttime < starttime:
                    continue

            if endtime is not None:
                if event.endtime > endtime:
                    continue

            events.append(event)
        return self.__class__(events=events)

    def load_distances(self, fnam_csv, overwrite=False):
        for event in self:
            event.load_distance_manual(fnam_csv,
                                       overwrite=overwrite)

    def calc_spectra(self, winlen_sec: float, detick_nfsamp=0) -> None:
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
        for event in tqdm(self, file=stdout):
            event.calc_spectra(winlen_sec=winlen_sec,
                               detick_nfsamp=detick_nfsamp)

    def save_magnitudes(self, fnam, version='Giardini2020', verbose=False):
        mags = []
        for event in self:
            mags.append([event.name,
                         event.magnitude(mag_type='mb_P', version=version, verbose=verbose),
                         event.magnitude(mag_type='mb_S', version=version, verbose=verbose),
                         event.magnitude(mag_type='m2.4', version=version, verbose=verbose),
                         event.magnitude(mag_type='MFB', version=version, verbose=verbose)
                         ])
        np.savetxt(fnam, mags, fmt=('%s'))

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
        for event in tqdm(self, file=stdout):
            event.read_waveforms(inv=inv, kind=kind, sc3dir=sc3dir,
                                 event_tmp_dir=event_tmp_dir)

    def plot_pickdiffs(
         self, pick1_X, pick2_X, pick1_Y, pick2_Y, vX=None, vY=None, fig=None,
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2'},
         markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
         markerfill={'A': True, 'B': True, 'C': False, 'D': False},
         show=True):

        if fig is None:
            fig = plt.figure()

        for event_type in ['2.4_HZ', 'HIGH_FREQUENCY']:
            for quality in 'ABCD':
                cat = self.select(quality=quality, event_type=event_type)
                times_X = []
                times_Y = []
                names = []
                S = []

                for event in cat:
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
                        S.append(markersize[event.quality])

                if len(names) == 0:
                    continue

                if vX is not None:
                    times_X = np.asarray(times_X) * vX
                if vY is not None:
                    times_Y = np.asarray(times_Y) * vY

                if markerfill[quality]:
                    colorargs = {'c': colors[event_type]}
                else:
                    colorargs = {'edgecolors': colors[event_type],
                                 'facecolor': 'none'}

                fig, ax = scatter_annot(times_X, times_Y, s=S, fig=fig,
                                        names=names,
                                        label=f'{event_type}, {quality}',
                                        **colorargs)
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


    def plot_pickdiff_over_time(
         self, pick1_Y, pick2_Y, vY=None, fig=None,
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2'},
         markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
         markerfill={'A': True, 'B': True, 'C': False, 'D': False},
         show=True):

        if fig is None:
            fig = plt.figure()

        for event_type in ['2.4_HZ', 'HIGH_FREQUENCY']:
            for quality in 'ABCD':
                cat = self.select(quality=quality, event_type=event_type)
                times_X = []
                times_Y = []
                names = []
                S = []
                for event in cat:
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
                        S.append(markersize[event.quality])

                if len(names) == 0:
                    continue

                if vY is not None:
                    times_Y = np.asarray(times_Y) * vY

                if markerfill[quality]:
                    colorargs = {'c': colors[event_type]}
                else:
                    colorargs = {'edgecolors': colors[event_type],
                                 'facecolor': 'none'}

                fig, ax = scatter_annot(times_X, times_Y, s=S, fig=fig,
                                        names=names,
                                        label=f'{event_type}, {quality}',
                                        **colorargs)

        ax.set_xlabel('Sol')
        if vY is None:
            ax.set_ylabel('$T_{%s} - T_{%s}$' % (pick1_Y, pick2_Y))
        else:
            ax.set_ylabel('distance / km (from %s-%s)' % (pick1_Y, pick2_Y))

        if show:
            plt.show()

    def plot_24_alignment(
         self, pre_time=120., post_time=120., fmax_filt=2.7, fmin_filt=2.1,
         envelope_window=100., amp_fac=2., show_picks=True, shift_to_S=False,
         regular_spacing=True, spacing=1., label=True, fill=True,
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2',
                 'VERY_HIGH_FREQUENCY': 'C3'},
         linestyle={'A': '-', 'B': '-', 'C': '--', 'D': ':'}, legend=True,
         #llabels=['VERY_HIGH_FREQUENCY', 'HIGH_FREQUENCY', '2.4_HZ'],
         linewidth=None, fig=None, cax=None, show=True):

        events = []
        for event in self:
            # filter for HF and 2.4 events
            if event.mars_event_type not in ['2.4_HZ', 'HIGH_FREQUENCY',
                                             'VERY_HIGH_FREQUENCY']:
                continue

            # Remove events that do not have all picks
            try:
                for pick in ['Pg', 'Sg', 'start', 'end', 'noise_start', 'noise_end']:
                    assert not event.picks[pick] == ''
            except:
                print('One or more picks missing for event %s' % (event.name))
            else:
                events.append(event)

        llabels = []
        for et in ['2.4_HZ', 'HIGH_FREQUENCY', 'VERY_HIGH_FREQUENCY']:
            if et in [e.mars_event_type for e in events]:
                llabels.append(et)

        # compute TP - TS to sort by distance
        tt_PgSg = np.array([(utct(event.picks['Sg']) -
                             utct(event.picks['Pg'])) for event in events])
        sorted_ids = np.argsort(tt_PgSg)

        if fig is None:
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
                start_shift = -tt_PgSg[i] if shift_to_S else 0.

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
            if regular_spacing:
                Y0 = k * spacing
                #Y0 = 0.
            else:
                Y0 = tt_PgSg[i]
                #Y0 = 0.

            if regular_spacing and spacing == 0.:
                Y = trZ_env.data * amp_fac
            else:
                Y = (trZ_env.data - np.median(trZ_noise_env.data)) * amp_fac + Y0
            #Y = (np.log10(trZ_env.data) -
            #     np.log10(np.median(trZ_noise_env.data))) * amp_fac + Y0
            #Y = np.log10(trZ_env.data) * amp_fac + Y0
            #Y = trZ_env.data * amp_fac + Y0

            # downsample to speed up plotting
            X = X[::10]
            Y = Y[::10]

            if spacing == 0.:
                color = pl.cm.jet((tt_PgSg[i] - tt_PgSg.min()) / tt_PgSg.ptp())
            else:
                color = colors[event.mars_event_type]
            plt.plot(X, Y, color=color,
                     ls=linestyle[event.quality], zorder=1000-k, lw=linewidth)

            if fill:
                # fill between noise amplitude estimate and envelope
                plt.fill_between(X, Y0, Y,  where=((Y>=Y0) * (X>start_shift) *
                                                   (X<duration+start_shift)),
                                 color=colors[event.mars_event_type], alpha=0.2,
                                 zorder=-20)

            if show_picks:
                if shift_to_S:
                    plt.plot([-tt, -tt], [Y0, Y0+0.3*amp_fac], color='C8')
                else:
                    plt.plot([tt, tt], [Y0, Y0+0.3*amp_fac], color='C8')
                plt.plot([duration + start_shift, duration + start_shift],
                         [Y0, Y0+0.8*amp_fac], color='C9')

            # # plot noise
            # X = trZ_noise_env.times()
            # X = X - pre_time - 400 - X[-1]
            # Y = trZ_noise_env.data * amp_fac + k
            # X = X[::10]
            # Y = Y[::10]
            # plt.plot(X, Y, color='k')

            if label:
                plt.text(-pre_time + start_shift, Y0, # + 0.5*amp_fac,
                         event.name + ' ',
                         ha='right', va='center')

        # time 0 line
        plt.axvline(0, color='k', ls='--')


        if not regular_spacing:
            plt.plot([-50, -400], [50, 400], color='k', ls='--')

            #plt.plot([-50, -400], [50, 400], color='C5')
            #plt.plot([-200, -400], [200, 400], color='C5')
            #bla = 0.5
            #plt.plot([-200 * bla, -400 * bla], [200, 400], color='C6')
            #bla = 0.5
            #plt.plot([200 * bla, 400 * bla], [200, 400], color='C6')
            #plt.plot([150, 150], [50, 400], color='C7')

        if legend:
            # legend
            lcolors = [colors[l] for l in llabels]
            llines = [Line2D([0], [0], color=c) for c in lcolors]
            plt.legend(llines, [EVENT_TYPES_PRINT[l] for l in llabels])

        # lable, limit, ticks
        plt.xlabel(f'time after {"S" if shift_to_S else "P"}g / s')
        plt.xlim(-pre_time - 300, None)
        #plt.yticks([], [])

        if regular_spacing and spacing == 0.:
            import matplotlib as mpl

            #plt.tick_params(axis='y', which='both', left=False, right=False,
            #                labelleft=False)

            cmap = mpl.cm.jet
            norm = mpl.colors.Normalize(vmin=tt_PgSg.min(), vmax=tt_PgSg.max())

            if cax is None:
                cax = fig.add_axes([0.35, 0.92, 0.6, 0.03])
                orientation = 'horizontal'
            else:
                orientation = 'vertical'

            cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                            norm=norm,
                                            orientation=orientation)
            cb1.set_label('Pg - Sg time / s')



        if show:
            plt.show()
        else:
            return fig

    def plot_HF_spectra(self, SNR=2., tooltip=False, component='Z', fmin=0.7,
                        quality='B', event_type=['2.4_HZ', 'HIGH_FREQUENCY',
                                                 'VERY_HIGH_FREQUENCY'],
                        fmax=10., use_SP=False, fig=None, show=True):
        from mpldatacursor import datacursor
        if fig is None:
            fig = plt.figure()
        ax = fig.gca()

        cat = self.select(quality=quality, event_type=event_type)

        class ContinueI(Exception):
            pass

        for event in cat:

            if use_SP:
                spectra = event.spectra_SP
            else:
                spectra = event.spectra

            # Skip events that do not have all picks, but print message in case
            try:
                for stype in ['P', 'S', 'noise']:
                    if not stype in spectra:
                        raise ContinueI(f'Missing spectral {stype} picks in event {event.name}')

                    if len(event.spectra[stype]) == 0:
                        raise ContinueI(f'Spectrum empty for {stype} in event {event.name}')
            except ContinueI as e:
                print(e)
                continue

            lw = 1.

            # mask_P_1Hz = (spectra['P'][f'f'] > 0.86) * # (spectra['P'][f'f'] < 1.14)
            mask_P_1Hz = spectra['P'][f'f'] > 1000.

            mask_P = spectra['P'][f'f'] < 1.3
            mask_P += spectra['P'][f'f'] > fmax
            peak = spectra['P'][f'p_{component}'][np.logical_not(mask_P)].max()
            mask_P = spectra['P'][f'f'] < fmin
            mask_P += spectra['P'][f'f'] > fmax
            mask_P += mask_P_1Hz
            mask_P += spectra['P'][f'p_{component}'] < SNR * spectra['noise'][f'p_{component}']
            msP = np.ma.masked_where(mask_P, spectra['P'][f'p_{component}'])
            msPN = np.ma.masked_where(mask_P_1Hz, spectra['P'][f'p_{component}'])

            msP /= peak
            msPN /= peak

            l1, = plt.plot(spectra['P'][f'f'], 10 * np.log10(msP),
                           color='C0', alpha=1., label=f'{event.name}, P',
                           lw=lw)
            plt.plot(spectra['P'][f'f'], 10 * np.log10(msPN),
                     color='lightgray', zorder=-10, lw=lw,
                     label=f'{event.name}, P noise')

            #mask_S_1Hz = (spectra['S'][f'f'] > 0.86) * (spectra['S'][f'f'] < 1.14)
            mask_S_1Hz = spectra['S'][f'f'] > 1000.

            mask_S = spectra['S'][f'f'] < 1.3
            mask_S += spectra['S'][f'f'] > fmax
            peak = spectra['S'][f'p_{component}'][np.logical_not(mask_S)].max()
            mask_S = spectra['S'][f'f'] < fmin
            mask_S += spectra['S'][f'f'] > fmax
            mask_S += mask_S_1Hz
            mask_S += spectra['S'][f'p_{component}'] < SNR * spectra['noise'][f'p_{component}']
            msS = np.ma.masked_where(mask_S, spectra['S'][f'p_{component}'])
            msSN = np.ma.masked_where(mask_S_1Hz, spectra['S'][f'p_{component}'])

            msS /= peak
            msSN /= peak

            l2, = plt.plot(spectra['S'][f'f'], 10 * np.log10(msS),
                           color='C1', alpha=1., label=f'{event.name}, S',
                           lw=lw)
            plt.plot(spectra['S'][f'f'], 10 * np.log10(msSN),
                     color='lightgray', zorder=-10, lw=lw,
                     label=f'{event.name}, S noise')

        if tooltip:
            datacursor(formatter='{label}'.format)

        # plot lorenz with attenuation
        f = np.linspace(0.01, 10., 1000)
        f_c = 100.
        if component == 'Z':
            ampfac = 30.
            delta_A0 = 0.
        else:
            ampfac = 10.
            delta_A0 = 4.5

        spec1 = lorentz_att(f, A0=-11.5 + delta_A0, f0=2.4, tstar=0.1, fw=0.3,
                            ampfac=ampfac, f_c=f_c)
        spec2 = lorentz_att(f, A0=-8.5 + delta_A0, f0=2.4, tstar=0.2, fw=0.3,
                            ampfac=ampfac, f_c=f_c)
        spec3 = lorentz_att(f, A0=-13 + delta_A0, f0=2.4, tstar=0.05, fw=0.3,
                            ampfac=ampfac, f_c=f_c)
        spec4 = lorentz_att(f, A0=-2 + delta_A0, f0=2.4, tstar=0.4, fw=0.3,
                            ampfac=ampfac, f_c=f_c)
        l3, = plt.plot(f, spec1, color='k', label='t* = 0.1 s')
        l4, = plt.plot(f, spec2, color='k', ls='--', label='t* = 0.2 s')
        l5, = plt.plot(f, spec3, color='k', ls='-.', label='t* = 0.05 s')
        l6, = plt.plot(f, spec4, color='k', ls=':', label='t* = 0.4 s')

        llabels = ['Pg', 'Sg'] + [l.get_label() for l in [l5, l3, l4, l6]]
        plt.legend([l1, l2, l5, l3, l4, l6], llabels)
        plt.xlabel('frequency / Hz')
        plt.ylabel('Amplitude relative to 2.4 peak / dB')

        ax = plt.gca()
        ax.set_xscale('log')
        xmajor_locator = matplotlib.ticker.LogLocator(
            base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0), numdecs=4, numticks=None)
        ax.get_xaxis().set_major_locator(xmajor_locator)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.xlim(0.3, 10.)
        plt.ylim(-30., 7.)

        if show:
            plt.show()
        else:
            return fig

    def plot_amplitude_PgSg(
         self,
         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2',
                 'VERY_HIGH_FREQUENCY': 'C0'},
         markers={'2.4_HZ': 'o', 'HIGH_FREQUENCY': 'o',
                  'VERY_HIGH_FREQUENCY': '^'},
         markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
         markerfill={'A': True, 'B': True, 'C': False, 'D': False},
         fig=None, show=True):

        if fig is None:
            fig = plt.figure()

        legend_elements = []

        for event_type in ['2.4_HZ', 'HIGH_FREQUENCY', 'VERY_HIGH_FREQUENCY']:
            for quality in 'ABCD':
                cat = self.select(quality=quality, event_type=event_type)

                for e in cat.events:
                    if e.picks['Sg'] == '' or e.picks['Pg'] == '':
                        print(f'missing pick on event {e.name}')
                cat.events = [e for e in cat.events if not e.picks['Sg'] == ''
                              and not e.picks['Pg'] == '']

                if len(cat) == 0:
                    continue

                # collect properties for plotting
                A = np.array([
                    event.amplitudes['A_24'] for event in cat]).astype(float)

                tt = np.array([
                    float(utct(event.picks['Sg']) - utct(event.picks['Pg']))
                    for event in cat])

                S = np.array([markersize[event.quality] for event in cat])
                names = np.array([f'{event.name} {event.duration_s:.0f}' for event in cat])

                mask = np.logical_not(np.isnan(A))
                A = A[mask]
                tt = tt[mask]
                S = S[mask]
                names = names[mask]

                if markerfill[quality]:
                    colorargs = {'c': colors[event_type]}
                else:
                    colorargs = {'edgecolors': colors[event_type],
                                 'facecolor': 'none'}

                scatter_annot(tt, A, s=S, fig=fig, names=names,
                              marker=markers[event_type],
                              label=f'{EVENT_TYPES_PRINT[event_type]} Q{quality}',
                              **colorargs)


        plt.xlabel('TSg - TPg / s')
        plt.ylabel('A2.4 / dB')

        vs = CRUST_VS
        vp = CRUST_VP
        d1 = degrees2kilometers(3 * (1. / vs - 1. / vp), RADIUS_MARS)
        d2 = degrees2kilometers(50 * (1. / vs - 1. / vp), RADIUS_MARS)
        dist = np.linspace(d1, d2)
        plt.plot(dist, -219.0 * np.ones_like(dist), label='-219.0 dB',
                 color='C3')
        plt.plot(dist, -212.5 * np.ones_like(dist), label='-212.5 dB',
                 color='C3', ls='--')
        plt.legend()

        if show:
            plt.show()
        else:
            return fig

    def plot_snr_dist(
            self, mag_type='m2.4',
            colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2',
                    'VERY_HIGH_FREQUENCY': 'C0'},
            xlabel='distance / degree [vs = 2 km/s, vp/vs = 1.7]',
            markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
            markerfill={'A': True, 'B': True, 'C': False, 'D': False},
            fig=None, show=True):

        if fig is None:
            fig = plt.figure()

        legend_elements = []

        for event_type in ['2.4_HZ', 'HIGH_FREQUENCY', 'VERY_HIGH_FREQUENCY']:
            for quality in 'ABCD':
                cat = self.select(quality=quality, event_type=event_type)

                if len(cat) == 0:
                    continue

                # collect properties for plotting
                M, dist = np.array([
                    (  # event.magnitude(mag_type=mag_type,
                        # distance=event.distance),
                        calc_stalta(event, fmin=2.2, fmax=2.8),
                        event.distance) for event in cat]).T.astype(float)

                S = np.array([markersize[event.quality] for event in cat])
                names = np.array(
                    [f'{event.name} {event.duration_s:.0f}' for event in cat])

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

        dist = np.linspace(3, 50)

        plt.xlabel(xlabel)
        plt.ylabel('SNR of event')
        plt.legend()

        if show:
            plt.show()
        else:
            return fig


    # def plot_magnitude_distance(
    #         self, mag_type='m2.4',
    #         version='Giardini2020',
    #         colors={'2.4_HZ': 'C1', 'HIGH_FREQUENCY': 'C2',
    #                 'VERY_HIGH_FREQUENCY': 'C0'},
    #         markers={'2.4_HZ': 'o', 'HIGH_FREQUENCY': 'o',
    #                  'VERY_HIGH_FREQUENCY': '^'},
    #         xlabel=f'distance / degree [vs = {CRUST_VS:3.1f} km/s, vp/vs = {CRUST_VP / CRUST_VS:3.1f}]',
    #         markersize={'A': 100, 'B': 50, 'C': 25, 'D': 5},
    #         markerfill={'A': True, 'B': True, 'C': False, 'D': False},
    #         fig=None, show=True):

    #     if fig is None:
    #         fig = plt.figure()

    #     legend_elements = []

    #     for event_type in ['2.4_HZ', 'HIGH_FREQUENCY', 'VERY_HIGH_FREQUENCY']:
    #         for quality in 'ABCD':
    #             cat = self.select(quality=quality, event_type=event_type)

    #             if len(cat) == 0:
    #                 continue

    #             # collect properties for plotting
    #             M, Msigma, dist = np.array([
    #                 (*event.magnitude(mag_type=mag_type, distance=event.distance, version=version),
    #                  event.distance) for event in cat]).T.astype(float)

    #             S = np.array([markersize[event.quality] for event in cat])
    #             names = np.array([f'{event.name} {event.duration_s:.0f}' for event in cat])

    #             mask = np.logical_not(np.isnan(M))
    #             M = M[mask]
    #             dist = dist[mask]
    #             S = S[mask]
    #             names = names[mask]

    #             if markerfill[quality]:
    #                 colorargs = {'c': colors[event_type]}
    #             else:
    #                 colorargs = {'edgecolors': colors[event_type],
    #                              'facecolor': 'none'}

    #             scatter_annot(dist, M, s=S, fig=fig, names=names,
    #                           marker=markers[event_type],
    #                           label=f'{EVENT_TYPES_PRINT[event_type]} Q{quality}',
    #                           **colorargs)

    #     dist = np.linspace(3, 50)
    #     magc_24 = M2_4(-219, dist)
    #     magc_HF = M2_4(-212.5, dist)
    #     plt.plot(dist, magc_24, label='M2.4(-219.0 dB)', color='C3')
    #     plt.plot(dist, magc_HF, label='M2.4(-212.5 dB)', color='C3', ls='--')

    #     plt.xlabel(xlabel)
    #     plt.ylabel('M2.4')
    #     plt.legend()

    #     if show:
    #         plt.show()
    #     else:
    #         return fig

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

    def plot_distance_distribution_cumulative(self, fig=None, label=None,
                                              show=True):

        if fig is None:
            fig = plt.figure()
        ax = fig.gca()

        d = np.array(sorted([e.distance for e in self]))
        N = np.arange(len(d)) / len(d)
        ax.plot(d, N, label=label)

        ax.set_xlabel('distance / degree')
        ax.set_ylabel('cumulative relative distribution of events')

        if label is not None:
            ax.legend()

        if show:
            plt.show()
        else:
            return fig

    def plot_distance_distribution_density(
         self, fig=None,
         xlabel=f'distance / degree [vs = {CRUST_VS:3.1f} km/s, vp/vs = {CRUST_VP/CRUST_VS:3.1f}]',
         label=None, show=True, color=None, plot_event_marker=True):

        if fig is None:
            fig = plt.figure()
        ax = fig.gca()

        d = np.array(sorted([e.distance for e in self]))
        print(np.mean(d), np.std(d))

        if plot_event_marker:
            plt.plot(d, np.zeros(d.shape), '|', ms=20, color=color)

        # kde_factor = len(d) ** (-0.2)  # Scott's rule
        kde = stats.gaussian_kde(d)
        x = np.linspace(0., 50., 1000)
        pdf1 = kde(x)
        plt.plot(x, pdf1, color=color, label=label)

        kde = stats.gaussian_kde(d, weights=1./d**2)
        x = np.linspace(0., 50., 1000)
        pdf1 = kde(x)
        plt.plot(x, pdf1, color=color, label=label + ' (area weighted)', ls='--')

        # kde = stats.gaussian_kde(np.log10(d), weights=1./d**2)
        # x = np.linspace(1., 50, 1000.)
        # pdf1 = kde(np.log10(x))
        # pdf1 = pdf1 / pdf1.sum() * 20
        # plt.plot(x, pdf1, color=color, label=label, ls=':')

        plt.xlabel(xlabel)
        plt.ylabel('PDF estimate from Gaussian KDE')

        if label is not None:
            ax.legend()

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
        for event in tqdm(self, file=stdout):
            for chan in ['Z', 'N', 'E']:
                fnam_report = pjoin(dir_out,
                                    'mag_report_%s_%s' %
                                    (event.name, chan))
                if not pexists(fnam_report + '.html'):
                    event.make_report(fnam_out=fnam_report,
                                      chan=chan,
                                      annotations=annotations)
                else:
                    event.fnam_report[chan] = fnam_report

    def make_report_parallel(self,
                             dir_out: str = 'reports',
                             annotations: Annotations = None):
        """
        Create Magnitude report figure
        :param dir_out: Directory to write report to
        :param annotations: Annotations object; used to mark glitches,
                            if available
        """
        from multiprocessing import Pool
        pool = Pool(processes=4)
        func = make_report_check_exists
        # func = print
        jobs = [pool.apply_async(func=func,
                                 args=(event,),
                                 kwds=dict(dir_out=dir_out,
                                           annotations=annotations))
                for event in self
                ]
        pool.close()
        result_list_tqdm = []
        for job in tqdm(jobs, file=stdout):
            result_list_tqdm.append(job.get())

        for event_name, fnam in result_list_tqdm:
            self.select(name=event_name).events[0].fnam_report = fnam

    def plot_filterbanks(self,
                         dir_out: str = 'filterbanks',
                         annotations: Annotations = None,
                         instrument: str = None,
                         fmax_LF: float = 8.,
                         fmin_LF: float = 1. / 32.,
                         fmax_HF: float = 16.,
                         fmin_HF: float = 1. / 2.,
                         df_LF: float = 2. ** 0.5,
                         df_HF: float = 2. ** 0.25
                         ):

        for event in tqdm(self, file=stdout):
            if event.mars_event_type_short in ['LF', 'BB']:
                if instrument is None:
                    instrument = 'VBB'
                if len(event.picks['S']) * len(event.picks['P']) > 0:
                    t_S = utct(event.picks['S'])
                    t_P = utct(event.picks['P'])
                else:
                    t_P = utct(event.starttime)
                    t_S = None
                fmin = fmin_LF
                fmax = fmax_LF
                df = df_LF
            elif event.mars_event_type_short in ['HF', '24']:
                if instrument is None:
                    instrument = 'SP'
                if len(event.picks['Sg']) * len(event.picks['Pg']) > 0:
                    t_S = utct(event.picks['Sg'])
                    t_P = utct(event.picks['Pg'])
                else:
                    t_P = utct(event.starttime)
                    t_S = None
                fmin = fmin_HF
                fmax = fmax_HF
                df = df_HF

            elif event.mars_event_type_short == 'VF':
                if event.available_sampling_rates()['SP_Z'] == 100.:
                    if instrument is None:
                        instrument = 'both'
                    if len(event.picks['Sg']) * len(event.picks['Pg']) > 0:
                        t_S = utct(event.picks['Sg'])
                        t_P = utct(event.picks['Pg'])
                    else:
                        t_P = utct(event.starttime)
                        t_S = None
                    fmin = 1./8.
                    fmax = 32.0 * np.sqrt(2.)
                    df = df_HF
                else:
                    if instrument is None:
                        instrument = 'SP'
                    if len(event.picks['Sg']) * len(event.picks['Pg']) > 0:
                        t_S = utct(event.picks['Sg'])
                        t_P = utct(event.picks['Pg'])
                    else:
                        t_P = utct(event.starttime)
                        t_S = None
                    fmin = 1./8.
                    fmax = 10.
                    df = df_HF

            else: # Super High Frequency
                if instrument is None:
                    instrument = 'SP'
                t_P = utct(event.starttime)
                t_S = None
                fmin = 0.5
                fmax = 32.0 * np.sqrt(2.)
                df = df_HF

            fnam = pjoin(dir_out, event.mars_event_type_short,
                         'filterbank_%s_all.png' % event.name)
            nodata = True
            if not pexists(fnam):
                try:
                    event.plot_filterbank(normwindow='all', annotations=annotations,
                                          starttime=event.starttime - 300.,
                                          endtime=event.endtime + 300.,
                                          instrument=instrument,
                                          fnam=fnam, fmin=fmin, fmax=fmax, df=df)
                except IndexError as err:
                    print(f'Problem with filterbank for event {event.name}')
                    print(err)
                except AttributeError as err:
                    print(f'Problem with filterbank for event {event.name}')
                    print(err)
                else:
                    nodata = False

            if event.quality in ['A', 'B', 'C'] and not nodata:
                fnam = pjoin(dir_out, event.mars_event_type_short,
                             'filterbank_%s_zoom.png' % event.name)
                try:
                    if not pexists(fnam):
                        event.plot_filterbank(starttime=t_P - 300.,
                                              endtime=t_P + 1100.,
                                              normwindow='S',
                                              annotations=annotations,
                                              tmin_plot=-240., tmax_plot=900.,
                                              fnam=fnam,
                                              instrument=instrument,
                                              fmin=fmin, fmax=fmax, df=df)

                    if t_S is not None:
                        fnam = pjoin(dir_out, event.mars_event_type_short,
                                     'filterbank_%s_phases.png' % event.name)
                        if not pexists(fnam):
                            event.plot_filterbank(starttime=t_P - 120.,
                                                  endtime=t_S + 240.,
                                                  normwindow='S',
                                                  annotations=annotations,
                                                  tmin_plot=-50.,
                                                  tmax_plot=t_S - t_P + 200.,
                                                  fnam=fnam,
                                                  instrument=instrument,
                                                  fmin=fmin, fmax=fmax, df=df)
                except IndexError as err:
                    print(f'Problem with filterbank for event {event.name}')
                    print(err)
            plt.close()

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
        nevents_LF = len(self.select(event_type=['LF', 'BB']))
        nevents_HF = len(self.select(event_type=['HF', '24', 'VF']))
        nrows_HF = max(1, (nevents_HF + 1) // 2)
        nrows_LF = max(2, (nevents_LF + 1) // 2)
        nrows = nrows_HF + nrows_LF + 1
        print('nevents:   ', nevents)
        print('nevents_LF:', nevents_LF)
        print('nevents_HF:', nevents_HF)
        print('nrow_LF:', nrows_LF)
        print('nrow_HF:', nrows_HF)
        hr = []
        for i in range(0, nevents_HF // 2):
            hr.append(2)
        hr.append(1)
        for i in range(0, nevents_LF // 2):
            hr.append(2)
        print(hr)
        fig, ax = plt.subplots(nrows=nrows, ncols=6, figsize=(14, 10),
                               sharex='all', sharey='all',
                               gridspec_kw={'height_ratios': hr})
        fig_all, ax_all = plt.subplots(nrows=1, ncols=2,
                                       sharex='all', sharey='all',
                                       figsize=(12, 6))
        self.select(event_type=['HF', '24', 'VF']).plot_many_spectra(
            ax, ax_all, df_mute, fits, nevents_HF, nrows_HF,
            source=False, iaxoff=0)
        self.select(event_type=['HF', '24', 'VF']).plot_many_spectra(
            ax, ax_all, df_mute, fits, nevents_HF, nrows_HF,
            iaxoff=0)
        self.select(event_type=['LF', 'BB']).plot_many_spectra(
            ax, ax_all, df_mute, fits, nevents_LF, nrows_LF,
            iaxoff=nrows_HF + 1)

        self.select(event_type=['LF', 'BB']).plot_many_spectra(
            ax, ax_all, df_mute, fits, nevents_LF, nrows_LF,
            source=False, iaxoff=nrows_HF + 1)
        # The subplots that are abused for text
        for ax_param in ax[:, [2, -1]].flatten():
            ax_param.set_frame_on(True)
            ax_param.tick_params(axis=u'both', which=u'both', length=0)
            ax_param.patch.set_visible(False)
            plt.setp(ax_param.get_xticklabels(), visible=False)
            for sp in ax_param.spines.values():
                sp.set_visible(False)
            pass
        # The subplots that act as spacing between HF and LF
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
        plt.close(fig_all)

        ax[0][0].set_xscale('log')
        ax[0][0].set_yticks((-140, -160, -180, -200, -220, -240))
        ax[0][0].set_yticklabels((-140, -160, -180, -200, -220, -240))
        ax[0][0].set_xlim(0.1, 8)
        ax[0][0].set_ylim(ymin, ymax)
        ax[0][0].set_title('vertical', fontsize=18)
        ax[0][1].set_title('horizontal', fontsize=18)
        ax[0][3].set_title('vertical', fontsize=18)
        ax[0][4].set_title('horizontal', fontsize=18)
        ax[nrows_HF + 1][0].set_title('vertical', fontsize=18)
        ax[nrows_HF + 1][1].set_title('horizontal', fontsize=18)
        ax[nrows_HF + 1][3].set_title('vertical', fontsize=18)
        ax[nrows_HF + 1][4].set_title('horizontal', fontsize=18)
        string = 'displacement PSD / (m$^2$/Hz) [dB]'
        ax[(nrows_HF) // 2][0].set_ylabel(string, fontsize=13)
        ax[nrows_HF + (nrows_LF + 1) // 2][0].set_ylabel(string,
                                                         fontsize=13)
        for a in [ax[-1][0], ax[-1][1], ax[-1][3], ax[-1][4]]:
            a.set_xlabel('frequency / Hz', fontsize=12)
        ax[-1][0].legend()  # bbox_to_anchor=(-0.4, 0.2))
        fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.985,
                            hspace=0.03, wspace=0.03)

        for ax_param in ax[nrows_HF, :].flatten():
            ax_param.set_frame_on(True)
            ax_param.tick_params(axis=u'both', which=u'both', length=0)
            ax_param.patch.set_visible(False)
            plt.setp(ax_param.get_xticklabels(), visible=False)
            for sp in ax_param.spines.values():
                sp.set_visible(False)
            pass

        fig.savefig('spectra_many_events.pdf')
        plt.show()

    def plot_many_spectra(self, ax, ax_all, df_mute, fits, nevents, nrows,
                          source=True, iaxoff=0):

        def f_c(M0, vs, ds):
            # Calculate corner frequency for event with M0,
            # assuming a stress drop ds
            return 4.9e-1 * vs * (ds / M0) ** (1 / 3)

        def M0(Mw):
            return 10 ** (Mw * 1.5 + 9.1)

        for a in ax_all:
            a.set_prop_cycle(plt.cycler('color',
                                        plt.cm.tab20(
                                            np.linspace(0, 1, nevents))))
        iax = iaxoff
        second = 0
        dists = []
        for event in self:
            if event.distance is not None:
                dists.append(event.distance)
            else:
                dists.append(30)
        order = np.argsort(dists)
        ievent = 0
        with open('time_windows_spectra.txt', 'a') as fid:
            for event in np.asarray(self.events)[order]:
                fid.write('%s, ' % event.name)
                ichan = 0
                if iax == nrows + iaxoff:
                    iax -= nrows
                    ichan = 3
                    second = 1
                if second == 1:
                    ichan = 3
                print('iax', iax, ' ichan', ichan, 'nrows', nrows, 'event', \
                      event.name)
                bodywave = False
                spectrum = event.spectra['noise']
                phase = fits[event.name]['phase']
                if len(spectrum) > 0:
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                                  spectrum, fmax=8., color='k', lw=2,
                                  label='noise')
                if 'S' in event.spectra:  # len(spectrum) > 0:
                    spectrum = event.spectra['S']
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                                  spectrum, fmax=8., color='r', lw=2,
                                  label='event')
                    # phase = 'S'
                    bodywave = True
                    fid.write('%s, ' % event.picks['S_spectral_start'])
                    fid.write('%s, ' % event.picks['S_spectral_end'])
                if 'P' in event.spectra and not bodywave:  # len(spectrum) > 0:
                    spectrum = event.spectra['P']
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                                  spectrum, fmax=8., color='r', lw=2,
                                  label='event')
                    # phase = 'P'
                    bodywave = True
                    fid.write('%s, ' % event.picks['P_spectral_start'])
                    fid.write('%s, ' % event.picks['P_spectral_end'])
                if 'all' in event.spectra and not bodywave:
                    spectrum = event.spectra['all']
                    plot_spectrum(ax, ax_all, df_mute, iax, ichan,
                                  spectrum, fmax=8., color='r', lw=2,
                                  label='total')
                    fid.write('%s, ' % event.picks['start'])
                    fid.write('%s, ' % event.picks['end'])
                    # phase = 'S'

                fid.write('%s, ' % event.picks['noise_start'])
                fid.write('%s, ' % event.picks['noise_end'])

                fid.write('\n')
                # if len(event.spectra_SP) > 0:
                #     if 'noise' in event.spectra_SP:
                #         spectrum = event.spectra_SP['noise']
                #         if len(spectrum) > 0:
                #             plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                #                           fmin=7., color='k')  # , label='noise')
                #     if 'P' in event.spectra_SP:
                #         spectrum = event.spectra_SP['P']
                #         if len(spectrum) > 0:
                #             plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                #                           fmin=7., color='b')  # , label='P-coda')
                #             bodywave = True
                #     if 'S' in event.spectra_SP:
                #         spectrum = event.spectra_SP['S']
                #         if len(spectrum) > 0:
                #             plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                #                           fmin=7., color='g')  # , label='S-code')
                #             bodywave = True
                #     spectrum = event.spectra_SP['all']
                #     if len(spectrum) > 0 and not bodywave:
                #         plot_spectrum(ax, ax_all, df_mute, iax, ichan, spectrum,
                #                       fmin=7., color='r')  # , label='total')

                if fits is not None:
                    if event.distance is None:
                        distance = 1600e3
                    else:
                        distance = event.distance * 55.e3
                    f = np.geomspace(0.01, 20., 100)
                    Mw = event.magnitude(mag_type='MFB')[0]
                    if Mw is None:
                        Mw = 3.
                    A0 = fits[event.name]['A0'] if 'A0' in fits[event.name] \
                        else event.amplitudes['A0']

                    # phase = fits[event.name]['phase']
                    p_pred = pred_spec(freqs=f,
                                       ds=1e6,
                                       mag=Mw,
                                       # amp=fits[event.name]['A0'],
                                       amp=A0,
                                       phase=phase,
                                       Qm=fits[event.name]['Qm'],
                                       dist=distance)
                    if source:
                        stf_amp = 20 * np.log10(1 / (1 + (f / f_c(M0=M0(Mw),
                                                                  vs=4.0e3,
                                                                  ds=1e5)) ** 2))
                        p_pred += stf_amp

                    ax[iax, ichan].plot(f, p_pred, c='darkblue', lw=2,
                                        ls='dashed',
                                        label='pred. src\n+ Att.')
                    ax[iax, ichan + 1].plot(f, p_pred, c='darkblue', lw=2,
                                            ls='dashed',
                                            label='pred. src\n+ Att.')
                    s = '$M_W$[$M^m_F$]=%3.1f\nPhase=%s\ndist=%d deg\n$Q_{' \
                        'eff}$=%d\n$A_0$=%ddB' \
                        % \
                        (Mw, phase, distance / 55e3, fits[event.name]['Qm'], A0)
                    ax[iax, ichan + 2].text(x=0.15, y=0.15, s=s,
                                            fontsize=10,
                                            transform=ax[iax,
                                                         ichan + 2].transAxes)

                ax[iax, ichan].text(x=0.96, y=0.96, s=event.name,
                                    fontsize=12, horizontalalignment='right',
                                    verticalalignment='top',
                                    bbox=dict(facecolor='white', alpha=0.5),
                                    transform=ax[iax, ichan].transAxes)

                iax += 1
                ievent += 1

    def write_table(self,
                    fnam_out: str = 'overview.html',
                    magnitude_version='Giardini2020') -> None:
        """
        Create HTML overview table for catalog
        :param fnam_out: filename to write to
        """
        from mqs_reports.create_table import write_html

        write_html(self, fnam_out=fnam_out, magnitude_version=magnitude_version)

    def get_event_count_table(self, style='html') -> str:
        """
        Create HTML event count table for catalog
        """

        import pandas as pd

        data = np.zeros((len(EVENT_TYPES), 5), dtype=int)

        for ie, event_type in enumerate(EVENT_TYPES):
            data[ie, 0] = len([e for e in self if e.mars_event_type == event_type])
            for iq, Q in enumerate('ABCD'):
                data[ie, iq+1] = len(
                    [e for e in self if (e.mars_event_type == event_type and
                                         e.quality == Q)])

        df = pd.DataFrame(data=data, columns=['total', 'A', 'B', 'C', 'D'])
        df.insert(loc=0, column='abbr.',
                  value=[f'{EVENT_TYPES_SHORT[e]}' for e in EVENT_TYPES])
        df.insert(loc=0, column='event type',
                  value=[f'{EVENT_TYPES_PRINT[e]}' for e in EVENT_TYPES])

        if style == 'html':
            return ('<H1>MQS events until %s</H1>\n<br>\n' %
                    utct().strftime('%Y-%m-%dT%H:%M (UTC)') +
                    df.to_html(index=False, table_id='events_all',
                               col_space=40)
                    )
        elif style == 'latex':
            return df.to_latex(index=False)
        else:
            raise ValueError()

    def plot_polarisation_analysis(self):
        """
        Create polarisation analysis plot
        """
        #Seconds before and after phase picks for signal window
        t_pick_P = [-5, 10]
        t_pick_S = [-5, 10]
        path_pol_plots = 'pol_plots'
        for event in tqdm(self):
            if (event.quality in ['A', 'B'] or \
                (event.quality == 'C' and
                event.mars_event_type_short in ['LF', 'BB'])) \
               and not pexists(
                       pjoin(path_pol_plots, f'{event.name}_polarisation.png')):
                baz=event.baz if event.baz else None
                for zoom in [False, True]:
                    try:
                        event.plot_polarisation(t_pick_P, t_pick_S,
                                                rotation_coords='ZNE', baz=baz,
                                                path_out=path_pol_plots,
                                                impact=False, zoom=zoom)
                    except ValueError as e:
                        print('Problem with event %s' % event.name)
                        print(e)
                        raise e

def make_report_check_exists(event, dir_out, annotations):
    fnam_report = dict()
    for chan in ['Z', 'N', 'E']:
        fnam_report[chan] = pjoin(dir_out,
                                  'mag_report_%s_%s' %
                                  (event.name, chan))
        if not pexists(fnam_report[chan] + '.html'):
            try:
                event.make_report(fnam_out=fnam_report[chan],
                                  chan=chan,
                                  annotations=annotations)
            except KeyError as e:
                print('Incomplete phases for event %s' % event.name)
                print(e)

    return event.name, fnam_report
