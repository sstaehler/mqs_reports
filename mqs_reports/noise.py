#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import obspy
from mars_tools.insight_time import solify
from matplotlib.patches import Polygon, Rectangle
from obspy import UTCDateTime as utct
from tqdm import tqdm

import mqs_reports
from mqs_reports.catalog import Catalog
from mqs_reports.utils import create_ZNE_HG, remove_sensitivity_stable

SECONDS_PER_DAY = 86400.


class Noise():
    def __init__(self,
                 data: dict = None,
                 sc3_dir: str = None,
                 starttime: obspy.UTCDateTime = None,
                 endtime: obspy.UTCDateTime = None,
                 inv: obspy.Inventory = None,
                 winlen_sec: float = None,
                 fmin_LF: float = 1. / 6.,
                 fmax_LF: float = 1. / 1.5,
                 fmin_HF: float = 2.2,
                 fmax_HF: float = 2.6,
                 fmin_HF_broad: float = 1.2,
                 fmax_HF_broad: float = 3.0,
                 fmin_press: float = 1. / 12.,
                 fmax_press: float = 1. / 1.5
                 ):
        self.sols_quant = None
        if data is None:
            self.sc3_dir = sc3_dir
            self.winlen_sec = winlen_sec

            self.stds_HF_broad = 0.
            self.stds_HF = 0.
            self.stds_LF = 0.
            self.stds_press = 0.
            self.times = 0.
            self.times_LMST = 0.
            self.sol = 0.
            self.fmin_LF = fmin_LF
            self.fmax_LF = fmax_LF
            self.fmin_HF = fmin_HF
            self.fmax_HF = fmax_HF
            self.fmin_HF_broad = fmin_HF_broad
            self.fmax_HF_broad = fmax_HF_broad
            self.fmin_press = fmin_press
            self.fmax_press = fmax_press
            self._add_data(starttime=starttime,
                           endtime=endtime,
                           inv=inv)

        else:
            self.stds_HF = np.asarray(data['stds_HF'])
            self.stds_LF = np.asarray(data['stds_LF'])
            self.stds_HF_broad = np.asarray(data['stds_HF_broad'])
            self.stds_press = np.asarray(data['stds_press'])
            self.times = np.asarray(data['times'])
            self.times_LMST = np.asarray(data['times_LMST'])
            self.sol = np.asarray(data['sol'])
            self.fmin_LF = data['freqs'][0]
            self.fmax_LF = data['freqs'][1]
            self.fmin_HF = data['freqs'][2]
            self.fmax_HF = data['freqs'][3]
            self.fmin_press = data['freqs'][4]
            self.fmax_press = data['freqs'][5]
            self.fmin_HF_broad = 1.2 #data['freqs'][6]
            self.fmax_HF_broad = 3.0 #data['freqs'][7]


    def __str__(self):
        fmt = 'Noise from %s to %s, time windows: %d(HF), %d(LF)'
        return fmt % (self.times[0].date, self.times[-1].date,
                      len(self.stds_LF), len(self.stds_HF))


    def _add_data(self,
                  starttime: obspy.UTCDateTime,
                  endtime: obspy.UTCDateTime,
                  inv: obspy.Inventory,
                  ):

        dirnam = pjoin(self.sc3_dir, 'op/data/waveform/%d/XB/ELYSE/BH?.D')
        filenam_VBB_HG = 'XB.ELYSE.0[23].BH?.D.%d.%03d'

        dirnam_pressure = pjoin(self.sc3_dir,
                                'op/data/waveform/%d/XB/ELYSE/?DO.D')

        jday_start = starttime.julday
        jday_end = int(float(endtime - starttime) / SECONDS_PER_DAY
                       + jday_start)
        year_start = starttime.year

        stds_HF = list()
        stds_LF = list()
        stds_press = list()
        stds_HF_broad = list()
        times = list()
        times_LMST = list()
        sol = list()
        print('reading noise seismic data from %s (%d-%d)' % (self.sc3_dir,
                                                              jday_start,
                                                              jday_end))
        for jday in tqdm(range(jday_start, jday_end)):
            year = year_start + (jday // 365)
            try:
                # TODO: This will fail in leap years
                fnam = pjoin(dirnam % year,
                             filenam_VBB_HG % (year, jday % 365))
                st = obspy.read(fnam)
            except Exception:
                print('did not find %s' % fnam)
                st = obspy.Stream()
            try:
                # Before switching to L2 continuous data
                if year == 2019 and jday < 120:
                    filenam_pressure = 'XB.ELYSE.02.MDO.D.%d.%03d'
                else:
                    filenam_pressure = 'XB.ELYSE.03.BDO.D.%d.%03d'
                # TODO: This will fail in leap years
                fnam_press = pjoin(dirnam_pressure % year,
                                   filenam_pressure % (year, jday % 365))
                st_press = obspy.read(fnam_press)
                if year > 2019 or jday > 120:
                    st_press.decimate(5)

            except Exception:
                print('did not find %s' % fnam)
                st_press = obspy.Stream()
            st.merge()
            if len(st.select(location='03')) == 3:
                st = st.select(location='03')
            else:
                st = st.select(location='02')
            if len(st) == 3:
                for tr in st:
                    remove_sensitivity_stable(tr, inv)
                st = create_ZNE_HG(st, inv=inv)
                st = st.select(channel='BHZ')
                st.filter('highpass', freq=1. / 10., corners=8)
                st.integrate()
                st.filter('highpass', freq=1. / 10., corners=8)

                st_filt_HF = st.copy()
                st_filt_HF.filter('highpass', freq=self.fmin_HF, corners=16)
                st_filt_HF.filter('lowpass', freq=self.fmax_HF, corners=16)

                st_filt_HF_broad = st.copy()
                st_filt_HF_broad.filter('highpass', freq=self.fmin_HF_broad, corners=16)
                st_filt_HF_broad.filter('lowpass', freq=self.fmax_HF_broad, corners=16)

                st_filt_LF = st.copy()
                st_filt_LF.filter('highpass', freq=self.fmin_LF, corners=16)
                st_filt_LF.filter('lowpass', freq=self.fmax_LF, corners=16)

                for tr in st_press:
                    remove_sensitivity_stable(tr, inv=inv)
                st_press.detrend()
                st_press.taper(max_length=10., max_percentage=1.)
                st_press.merge(method=0, fill_value='interpolate')
                st_press.filter('highpass', freq=self.fmin_press, corners=16)
                st_press.filter('highpass', freq=self.fmin_press, corners=16)
                st_press.filter('lowpass', freq=self.fmax_press, corners=16)

                for t in np.arange(0, SECONDS_PER_DAY, self.winlen_sec):
                    t0 = utct('%04d%03d' % (year, jday % 365)) + t
                    t1 = t0 + self.winlen_sec
                    st_win = st.slice(starttime=t0, endtime=t1)
                    st_filt_HF_broad_win = st_filt_HF_broad.slice(
                                                      starttime=t0,
                                                      endtime=t1)
                    st_filt_HF_win = st_filt_HF.slice(starttime=t0,
                                                      endtime=t1)
                    st_filt_LF_win = st_filt_LF.slice(starttime=t0,
                                                      endtime=t1)

                    st_press_win = st_press.slice(starttime=t0, endtime=t1)
                    if len(st_win) > 0 and st_win[0].stats.npts > 10:
                        std_HF_broad = st_filt_HF_broad_win[0].std()
                        std_HF = st_filt_HF_win[0].std()
                        std_LF = st_filt_LF_win[0].std()

                        if len(st_press_win) > 0 and \
                                st_press_win[0].stats.npts > 10:
                            std_press = st_press_win[0].std()
                        else:
                            std_press = 0.

                        stds_press.append(std_press)
                        stds_HF_broad.append(std_HF_broad)
                        stds_HF.append(std_HF)
                        stds_LF.append(std_LF)
                        t0_lmst = solify(t0)

                        times.append(t0)
                        times_LMST.append(float(t0_lmst) / SECONDS_PER_DAY)
                        sol.append(int(float(t0_lmst) / SECONDS_PER_DAY))

        self.stds_HF_broad = np.asarray(stds_HF_broad)
        self.stds_HF = np.asarray(stds_HF)
        self.stds_LF = np.asarray(stds_LF)
        self.stds_press = np.asarray(stds_press)
        self.times = np.asarray(times)
        self.times_LMST = np.asarray(times_LMST)
        self.sol = np.asarray(sol)

    def save(self, fnam):
        np.savez(fnam,
                 stds_HF_broad=self.stds_HF_broad,
                 stds_HF=self.stds_HF,
                 stds_LF=self.stds_LF,
                 stds_press=self.stds_press,
                 times=self.times,
                 times_LMST=self.times_LMST,
                 sol=self.sol,
                 freqs=[self.fmin_LF, self.fmax_LF,
                        self.fmin_HF, self.fmax_HF,
                        self.fmin_press, self.fmax_press,
                        self.fmin_HF_broad, self.fmax_HF_broad],
                 winlen_sec=self.winlen_sec)


    def plot_noise_stats(self, sol_start=80, sol_end=None,
                         ax=None, show=True):
        power_bins, p_LF, p_HF = self.calc_noise_stats(sol_end,
                                                       sol_start)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(power_bins, p_LF,
                label='LF, 1.5 - 6 seconds')
        ax.plot(power_bins, p_HF,
                label='HF, 2 - 3 Hz')
        plt.legend()

        if show:
            plt.show()

    def calc_noise_stats(self, sol_end=None, sol_start=80):
        if sol_end is None:
            # Now
            sol_end = float(solify(utct())) // SECONDS_PER_DAY
        binwidth = 2.
        bins = np.arange(-260, -120, binwidth)
        bol_LF = np.array([np.isfinite(self.stds_LF),
                           self.sol > sol_start,
                           self.sol < sol_end]).all(axis=0)
        power_LF, bins_tmp = np.histogram(
            20 * np.log10(self.stds_LF[bol_LF]),
            bins=bins, density=True)
        bol_HF = np.array([np.isfinite(self.stds_HF),
                           self.sol > sol_start,
                           self.sol < sol_end]).all(axis=0)
        power_HF, bins_tmp = np.histogram(
            20 * np.log10(self.stds_HF[bol_HF]),
            bins=bins, density=True)
        bins = bins[0:-1] + binwidth / 2.
        p_LF = np.cumsum(power_LF) * binwidth
        p_HF = np.cumsum(power_HF) * binwidth
        return bins, p_LF, p_HF

    def calc_noise_quantiles(self, qs,
                             sol_start=80, sol_end=None):
        if sol_end is None:
            # Now
            sol_end = float(solify(utct())) // 86400

        quantiles_HF = []
        quantiles_LF = []
        for q in qs:
            bol_LF = np.array([np.isfinite(self.stds_LF),
                               self.sol > sol_start,
                               self.sol < sol_end]).all(axis=0)
            quantiles_LF.append(
                np.quantile(a=20*np.log10(self.stds_LF[bol_LF]), q=q)
                )
            bol_HF = np.array([np.isfinite(self.stds_HF),
                               self.sol > sol_start,
                               self.sol < sol_end]).all(axis=0)
            quantiles_HF.append(
                np.quantile(a=20*np.log10(self.stds_HF[bol_HF]), q=q)
                )

        return np.asarray(quantiles_LF), np.asarray(quantiles_HF)


    def plot_daystats(self, cat: mqs_reports.catalog.Catalog = None,
                      sol_start: int = 80, sol_end: int = 500, data_apss=False,
                      fnam_out='noise_vs_eventamplitudes.png', extra_data=None,
                      cmap_dist='plasma_r', tau_data=None, metal=False):

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        if self.sols_quant is None:
            self.calc_time_windows(sol_end, sol_start)
            self.save_quantiles(fnam='noise_quantiles.npz')

        # verts_LF = []
        # verts_HF = []

        # for i, isol in enumerate(self.sols_quant):
        #     if self.quantiles_LF[i, 0] > 0:
        #         verts_LF.append([isol, 10 * np.log10(self.quantiles_LF[i, 0])])
        #     if self.quantiles_HF[i, 0] > 0:
        #         verts_HF.append([isol, 10 * np.log10(self.quantiles_HF[i, 0])])
        # verts_LF.append([verts_LF[-1][0], -300])
        # verts_HF.append([verts_HF[-1][0], -300])
        # verts_HF.append([self.sols_quant[0], -300])
        # verts_LF.append([self.sols_quant[0], -300])

        fig = plt.figure(figsize=(16, 9))
        # ax_HF = ax[1
        # ax_LF = ax[0]
        h_base = 0.09
        w_base = 0.05
        w_LF = 0.85
        h_pad = 0.06
        h_LF = (0.97 - h_base - h_pad) / 2.

        if data_apss:
            h_apss = 0.2
            h_LF = (0.97 - h_base - 2 * h_pad - h_apss) / 2.
            # h_LF -= h_apss / 2. + h_pad

        ax_HF = fig.add_axes([w_base, h_base,
                              w_LF, h_LF])
        if data_apss:
            ax_apss = fig.add_axes([w_base, h_base + h_LF * 2 + h_pad * 2,
                                    w_LF, h_apss],
                                   sharex=ax_HF)
            if extra_data is not None:
                ax_extra = ax_apss.twinx()
                m = 's'
                ax_extra.plot(extra_data[0], extra_data[1],
                              markeredgecolor='k', markerfacecolor='C0',
                              marker=m, ls=None, lw=0., ms=4.,
                              label='DuDes per Sol\n(A. Spiga)')
                ax_extra.set_ylim(0, 60)
                ax_extra.set_ylabel('number of DuDes')
                ax_extra.yaxis.label.set_color('C0')
                ax_extra.tick_params(axis='y', colors='C0')
                ax_extra.legend(loc='upper left')
                if tau_data is not None:
                    ax_tau = ax_apss.twinx()
                    # Offset the right spine of par2.  The ticks and label have
                    # already been
                    # placed on the right by twinx above.
                    ax_tau.spines["right"].set_position(("axes", 1.05))
                    # Having been created by twinx, par2 has its frame off, so
                    # the line of its
                    # detached spine is invisible.  First, activate the frame
                    # but make the patch
                    # and spines invisible.
                    make_patch_spines_invisible(ax_tau)
                    # Second, show the right spine.
                    ax_tau.spines["right"].set_visible(True)
                    m = 'o'
                    ax_tau.plot(tau_data[0], tau_data[1],
                                  markeredgecolor='k', markerfacecolor='orange',
                                  marker=m, ls=None, lw=0., ms=6.,
                                  label='tau\n(M. Lemmon)')
                    ax_tau.set_ylim(0, 1.2)
                    ax_tau.set_ylabel('tau')
                    ax_tau.yaxis.label.set_color('orange')
                    ax_tau.tick_params(axis='y', colors='orange')
                    ax_tau.legend()
        else:
            ax_apss = fig.add_axes([1.2, 1.2, 0.1, 0.1])

        ax_LF = fig.add_axes([w_base, h_base + h_LF + h_pad,
                              w_LF, h_LF],
                             sharex=ax_HF)

        cols = ['black', 'darkgrey', 'grey', 'black']
        ls = ['dotted', 'dashed', 'dotted', 'dashed']
        if metal: 
            labels = ['quiet\n(17-22 LMST)', 'NÖCTURNAL WAVEGÜIDE\n(22-5 LMST)',
                      'SÜNRÏSE\n(5-9 LMST)', 'SËISMIC NÖISË\n(9-17 LMST)']
        else:
            labels = ['quiet\n(17-22 LMST)', 'noisy night\n(22-5 LMST)',
                      'morning\n(5-9 LMST)', 'day\n(9-17 LMST)']
        for i in range(self.quantiles_LF.shape[1] - 1, -1, -1):
            ax_LF.plot(self.sols_quant - 1.,
                       10 * np.log10(self.quantiles_LF[:, i]),
                       label=labels[i], c=cols[i], ls=ls[i])
        ax_LF.fill_between(x=self.sols_quant - 1.,
                           y1=-300,
                           y2=10 * np.log10(self.quantiles_LF[:, 0]),
                           facecolor='lightgrey')

        for i in range(self.quantiles_HF.shape[1] - 1, -1, -1):
            ax_HF.plot(self.sols_quant - 1.,
                       10 * np.log10(self.quantiles_HF[:, i]),
                       label=labels[i], c=cols[i], ls=ls[i])
        ax_HF.fill_between(x=self.sols_quant - 1.,
                           y1=-300,
                           y2=10 * np.log10(self.quantiles_HF[:, 0]),
                           facecolor='lightgrey')
        if data_apss:
            for i in range(self.quantiles_press.shape[1] - 1, -1, -1):
                ax_apss.plot(self.sols_quant - 1.,
                             10 * np.log10(self.quantiles_press[:, i]),
                             label=labels[i], c=cols[i], ls=ls[i])
            ax_apss.fill_between(x=self.sols_quant - 1.,
                                 y1=-300,
                                 y2=10 * np.log10(self.quantiles_press[:, 0]),
                                 facecolor='lightgrey')
        HF_times = []
        HF_amps = []
        HF_dists = []
        LF_times = []
        LF_amps = []
        LF_dists = []
        symbols = {'2.4_HZ': '>',
                   'HIGH_FREQUENCY': 'v',
                   'VERY_HIGH_FREQUENCY': '^',
                   'LOW_FREQUENCY': 's',
                   'BROADBAND': 'D'}
        markers_HF = []
        markers_LF = []

        if cat is not None:
            cmap = plt.cm.get_cmap(cmap_dist)
            cmap.set_over('white')
            for event in cat.select(event_type=['HF', 'VF', '24']):
                markers_HF.append(symbols[event.mars_event_type])
                if event.distance is None:
                    HF_dists.append(50.)
                else:
                    HF_dists.append(event.distance)
                HF_times.append(float(solify(event.starttime)) /
                                SECONDS_PER_DAY)
                HF_amps.append(event.amplitudes['A_24'])

            for event in cat.select(event_type=['LF', 'BB']):
                markers_LF.append(symbols[event.mars_event_type])
                if event.distance is None:
                    LF_dists.append(120.)
                else:
                    LF_dists.append(event.distance)
                amp_P = event.pick_amplitude(
                    pick='Peak_MbP',
                    comp='vertical',
                    fmin=self.fmin_LF,
                    fmax=self.fmax_LF,
                    instrument='VBB'
                    )
                amp_S = event.pick_amplitude(
                    pick='Peak_MbS',
                    comp='vertical',
                    fmin=self.fmin_LF,
                    fmax=self.fmax_LF,
                    instrument='VBB'
                    )
                amp = max(i for i in (amp_P, amp_S, 0.0) if i is not None)
                LF_times.append(float(solify(event.starttime)) /
                                SECONDS_PER_DAY)
                LF_amps.append(20 * np.log10(amp))


            # for i, m in enumerate(np.unique(markers_LF)): 
            for event_type, m in symbols.items():
                bol = np.array(markers_LF) == m
                if len(bol) > 0:
                    sc = ax_LF.scatter(x=np.array(LF_times)[bol], 
                                       y=np.array(LF_amps)[bol],
                                       c=np.array(LF_dists)[bol],  
                                       vmin=15., vmax=100., cmap=cmap,
                                       edgecolors='k', linewidths=0.5,
                                       s=80., marker=m, zorder=100)
            cax = fig.add_axes([w_LF + w_base + 0.02, h_base + 0.1, 
                                0.018, 1 - h_base - 0.5])
            cb = plt.colorbar(sc, ax=ax_LF, cax=cax) # use_gridspec=True)
            cb.ax.set_ylabel('distance / degree', rotation=270.,
                              labelpad=4.45)
            # for i, m in enumerate(np.unique(markers_HF)): 
            for event_type, m in symbols.items():
                bol = np.array(markers_HF) == m
                if len(bol) > 0:
                    sc = ax_HF.scatter(x=np.array(HF_times)[bol], 
                                       y=np.array(HF_amps)[bol],
                                       c=np.array(HF_dists)[bol],  
                                       vmin=15., vmax=100., cmap=cmap,
                                       edgecolors='k', linewidths=0.5,
                                       s=40., marker=m, zorder=100)
            # cax = plt.colorbar(sc, ax=ax_HF, use_gridspec=True, 
            #                    fraction=0.08)
            # cax.ax.set_ylabel('distance / degree', rotation=270.,
            #                   labelpad=12.45)
            # if data_apss:
            #     cax = plt.colorbar(sc, ax=ax_apss, use_gridspec=True)
            #     cax.ax.set_ylabel('distance / degree', rotation=270.,
            #                       labelpad=12.45)

        for a in [ax_HF, ax_LF, ax_apss]:
            a.grid(True)
            rect = Rectangle(xy=(267, -300), width=20., height=400, zorder=10,
                             facecolor='darkgrey', edgecolor='black')
            a.add_patch(rect)

        sc = ax_HF.scatter(0, -300, label='Marsquake',
                           edgecolors='k', linewidths=0.5,
                           c='royalblue', s=80., marker='.')
        ax_HF.set_xlabel('Sol number')
        ax_LF.set_ylabel('PSD, displ. %3.1f-%3.1f sec. [dB]' %
                         (1. / self.fmax_LF, 1. / self.fmin_LF))
        ax_HF.set_ylabel('PSD, displ. %3.1f-%3.1f Hz. [dB]' %
                         (self.fmin_HF, self.fmax_HF))

        ax_LF.set_ylim(-210., -170.)
        ax_LF.set_title('Seismic power, low frequency,  %3.1f-%3.1f seconds and LF/BB events' %
                        (1. / self.fmax_LF, 1. / self.fmin_LF))
        ax_HF.set_title('Seismic power, high frequency,  %3.1f-%3.1f Hz and HF/2.4 Hz events' %
                        (self.fmin_HF, self.fmax_HF))
        ax_HF.set_ylim(-225., -185.)
        ax_LF.set_xlim(sol_start, sol_end)


        ax_HF.text(0.7, -0.12, s=str(self),
                   transform=ax_HF.transAxes)

        # Daytimes legend
        l = ax_LF.legend(bbox_to_anchor=(w_LF + w_base, 0.6, 0.1, 0.2),
                         bbox_transform=fig.transFigure, framealpha=1.0)
        l.set_zorder(50)

        handles = []
        labels = []
        for event_type, m in symbols.items():
            h, = ax_HF.plot(-100, -100, markeredgecolor='k', markerfacecolor='white', 
                            marker=m, ls=None, lw=0., ms=8.)
            handles.append(h)
            labels.append(event_type)
        l = ax_HF.legend(handles=handles, labels=labels,
                         bbox_to_anchor=(w_LF + w_base, h_base, 0.1, 0.1), 
                         bbox_transform=fig.transFigure, framealpha=1.0)
        l.set_zorder(50)

        if data_apss:
            ax_apss.set_title('Pressure power %3.1f-%3.1f seconds' %
                              (1. / self.fmax_press, 1. / self.fmin_press))
            ax_apss.set_ylabel('PSD, pressure.\n%3.1f-%4.1f sec. [dB]' %
                               (1. / self.fmax_press, 1. / self.fmin_press))
            ax_apss.set_ylim(-50, -25)

        plt.savefig(fnam_out, dpi=200)

    def read_quantiles(self, fnam):
        data = np.load(fnam)
        self.quantiles_HF = data['quantiles_HF']
        self.quantiles_LF = data['quantiles_LF']
        self.quantiles_press = data['quantiles_press']
        self.sols_quant = data['sols']

    def save_quantiles(self, fnam):
        np.savez(file=fnam,
                 quantiles_HF=self.quantiles_HF,
                 quantiles_LF=self.quantiles_LF,
                 quantiles_press=self.quantiles_press,
                 sols=self.sols_quant)

    def calc_time_windows(self, sol_end, sol_start,
                          time_windows_hour=[[17, 22.0],
                                             [22.0, 5.0],
                                             [5.0, 9.0],
                                             [9.0, 17.]]):

        self.sols_quant = np.arange(sol_start, sol_end + 1)
        self.quantiles_LF = np.zeros(
            (sol_end - sol_start + 1, len(time_windows_hour)))
        self.quantiles_HF = np.zeros_like(self.quantiles_LF)
        self.quantiles_press = np.zeros_like(self.quantiles_LF)

        values_HF = np.zeros(len(time_windows_hour))
        values_LF = np.zeros_like(values_HF)
        values_press = np.zeros_like(values_HF)

        i = 0
        print('Calculating noise medians')
        df_HF = self.fmax_HF - self.fmin_HF
        df_LF = self.fmax_LF - self.fmin_LF
        df_press = self.fmax_press - self.fmin_press
        for isol in tqdm(self.sols_quant):
            bol_sol = self.sol == isol - 1
            times_this_sol = (self.times_LMST[bol_sol] % 1.) * 24.
            if sum(bol_sol) > 1:
                for iwindow, time_window in enumerate(time_windows_hour):
                    if time_window[0] < time_window[1]:
                        bol_hours = np.array((
                            (time_window[0] < times_this_sol),
                            (times_this_sol < time_window[1]))).all(axis=0)
                    else:
                        bol_hours = np.array((
                            (time_window[0] < times_this_sol),
                            (times_this_sol < time_window[1]))).any(axis=0)

                    disp_LF = self.stds_LF[bol_sol] ** 2. / df_LF
                    disp_HF = self.stds_HF[bol_sol] ** 2. / df_HF
                    disp_press = self.stds_press[bol_sol] ** 2. / df_press
                    values_LF[iwindow] = np.nanmedian(disp_LF[bol_hours])
                    values_HF[iwindow] = np.nanmedian(disp_HF[bol_hours])
                    values_press[iwindow] = np.nanmedian(disp_press[bol_hours])

                self.quantiles_HF[i, :] = np.nan_to_num(values_HF)
                self.quantiles_LF[i, :] = np.nan_to_num(values_LF)
                self.quantiles_press[i, :] = np.nan_to_num(values_press)
            i += 1

        # self.quantiles_LF, self.quantiles_HF = self.calc_noise_quantiles(
        #         qs=[0.05, 0.25, 0.5, 0.9], sol_start=sol_start, sol_end=sol_end)

        # Mask outliers
        self.quantiles_LF = np.ma.masked_less(self.quantiles_LF, value=1e-23)
        self.quantiles_HF = np.ma.masked_less(self.quantiles_HF, value=1e-23)
        self.quantiles_press = np.ma.masked_less(self.quantiles_press,
                                                 value=1e-6)
        self.quantiles_LF = np.ma.masked_greater(self.quantiles_LF, value=1e-8)
        self.quantiles_HF = np.ma.masked_greater(self.quantiles_HF, value=1e-8)
        self.quantiles_press = np.ma.masked_greater(self.quantiles_press,
                                                    value=1e-2)


    def compare_events(self,
                       catalog=None,
                       threshold_dB: float = 3.):
        ratios = []
        for event in tqdm(catalog.select(event_type=['24', 'HF', 'VF'])):
            nwins_below = sum(event.amplitudes['A_24'] >
                              threshold_dB + 20 * np.log10(self.stds_HF))
            event.ratio = nwins_below / len(self.stds_HF)
            ratios.append(event.ratio)

        for event in tqdm(catalog.select(event_type=['LF', 'BB'])):
            amp_P = event.pick_amplitude(
                pick='Peak_MbP',
                comp='vertical',
                fmin=self.fmin_LF,
                fmax=self.fmax_LF,
                instrument='VBB'
                )
            amp_S = event.pick_amplitude(
                pick='Peak_MbS',
                comp='vertical',
                fmin=self.fmin_LF,
                fmax=self.fmax_LF,
                instrument='VBB'
                )
            if amp_S is None:
                amp = amp_P
            elif amp_P is None:
                amp = amp_S
            else:
                amp = max((amp_P, amp_S))
            if amp is not None:
                nwins_below = sum(20 * np.log10(amp) >
                                  threshold_dB + 20 * np.log10(self.stds_LF))
                event.ratio = nwins_below / len(self.stds_LF)
            else:
                event.ratio = None
            ratios.append(event.ratio)
        # import matplotlib.pyplot as plt

        # plt.hist(ratios, bins=20)
        # plt.show()


def read_noise(fnam):
    data = np.load(fnam, allow_pickle=True)
    data_read = dict()
    for name in data.files:
        data_read[name] = data[name]

    return Noise(data=data_read)


if __name__ == '__main__':
    sc3_path = '/mnt/mnt_sc3data'
    inv = obspy.read_inventory('mqs_reports/data/inventory.xml')
    # noise = Noise(sc3dir=sc3_path,
    #               starttime=utct('20190202'),
    #               endtime=utct('20191026'),
    #               inv=inv,
    #               winlen_sec=120.
    #               )
    # noise.save('noise_0301_1025.npz')
    noise = read_noise('noise_0301_1025.npz')
    noise.plot_noise_stats()
    noise.read_quantiles(fnam='quantiles.npz')

    cat = Catalog(fnam_quakeml='mqs_reports/data/catalog_20191024.xml',
                  quality=['A', 'B'])

    cat.load_distances(fnam_csv='./mqs_reports/data/manual_distances.csv')
    cat.read_waveforms(inv=inv, sc3dir=sc3_path)
    cat.calc_spectra(winlen_sec=10.)
    noise.plot_daystats(cat, sol_start=200, sol_end=300)
    noise.compare_events(cat)
    noise.plot_noise_stats()


def calc_Ls(t_LMST):
    import marstime
    from mars_tools.insight_time import UTCify
    t = float(UTCify(t_LMST)) * 1e3

    j2000_offset = marstime.j2000_offset_tt(
        marstime.julian_tt(marstime.julian(t)))
    ls = marstime.Mars_Ls(j2000_ott=j2000_offset)  # + 30 * 86400e3))))
    return ls
