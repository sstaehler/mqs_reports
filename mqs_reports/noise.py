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
import mqs_reports
import numpy as np
import obspy
from mars_tools.insight_time import solify
from matplotlib.patches import Polygon, Rectangle
from mqs_reports.catalog import Catalog
from mqs_reports.utils import create_ZNE_HG
from obspy import UTCDateTime as utct
from tqdm import tqdm


class Noise():
    def __init__(self,
                 data: dict = None,
                 sc3_dir: str = None,
                 starttime: obspy.UTCDateTime = None,
                 endtime: obspy.UTCDateTime = None,
                 inv: obspy.Inventory = None,
                 winlen_sec: float = None,
                 ):
        self.sols_quant = None
        if data is None:
            self.sc3_dir = sc3_dir
            self.winlen_sec = winlen_sec

            self.stds_HF = list()
            self.stds_LF = list()
            self.times = list()
            self.times_LMST = list()
            self.sol = list()
            self._add_data(starttime=starttime,
                           endtime=endtime,
                           inv=inv)

        else:
            self.stds_HF = np.asarray(data['stds_HF'])
            self.stds_LF = np.asarray(data['stds_LF'])
            self.times = np.asarray(data['times'])
            self.times_LMST = np.asarray(data['times_LMST'])
            self.sol = np.asarray(data['sol'])


    def __str__(self):
        fmt = 'Noise from %s to %s, time windows: %d(HF), %d(LF)'
        return fmt % (self.times[0].date, self.times[-1].date,
                      len(self.stds_LF), len(self.stds_HF))

    def _add_data(self,
                  starttime: obspy.UTCDateTime,
                  endtime: obspy.UTCDateTime,
                  inv: obspy.Inventory):

        dirnam = pjoin(self.sc3_dir, 'op/data/waveform/%d/XB/ELYSE/BH?.D')
        filenam_VBB_HG = 'XB.ELYSE.0[23].BH?.D.%d.%03d'

        jday_start = starttime.julday
        jday_end = endtime.julday
        year = starttime.year

        stds_HF = list()
        stds_LF = list()
        times = list()
        times_LMST = list()
        sol = list()
        print('reading seismic data from %s' % self.sc3_dir)
        for jday in tqdm(range(jday_start, jday_end)):
            try:
                fnam = pjoin(dirnam % year,
                             filenam_VBB_HG % (year, jday))
                st = obspy.read(fnam)
            except Exception:
                st = obspy.Stream()
            st.merge()
            if len(st.select(location='03')) == 3:
                st = st.select(location='03')
            else:
                st = st.select(location='02')
            if len(st) == 3:
                try:
                    st.remove_sensitivity(inv)
                except ValueError:
                    print('Inventory problem on jday %d' % jday)
                else:
                    st = create_ZNE_HG(st, inv=inv)
                    st = st.select(channel='BHZ')
                    st.filter('highpass', freq=1. / 10., corners=8)
                    st.integrate()
                    st.filter('highpass', freq=1. / 10., corners=8)

                    st_filt_HF = st.copy()
                    st_filt_HF.filter('highpass', freq=2.2, corners=16)
                    st_filt_HF.filter('lowpass', freq=2.6, corners=16)

                    st_filt_LF = st.copy()
                    st_filt_LF.filter('highpass', freq=1 / 6, corners=16)
                    st_filt_LF.filter('lowpass', freq=1 / 1.5, corners=16)

                    for t in np.arange(0, 86400, self.winlen_sec):
                        t0 = obspy.UTCDateTime('2019%03d' % jday) + t
                        t1 = t0 + self.winlen_sec
                        st_win = st.slice(starttime=t0, endtime=t1)
                        st_filt_HF_win = st_filt_HF.slice(starttime=t0,
                                                          endtime=t1)
                        st_filt_LF_win = st_filt_LF.slice(starttime=t0,
                                                          endtime=t1)
                        if len(st_win) > 0 and st_win[0].stats.npts > 10:
                            std_HF = st_filt_HF_win[0].std()
                            std_LF = st_filt_LF_win[0].std()
                            stds_HF.append(std_HF)
                            stds_LF.append(std_LF)
                            t0_lmst = solify(t0)

                            times.append(t0)
                            times_LMST.append(float(t0_lmst) / 86400.)
                            sol.append(int(float(t0_lmst) // 86400))

        self.stds_HF = np.asarray(stds_HF)
        self.stds_LF = np.asarray(stds_LF)
        self.times = np.asarray(times)
        self.times_LMST = np.asarray(times_LMST)
        self.sol = np.asarray(sol)

    def save(self, fnam):
        np.savez(fnam,
                 stds_HF=self.stds_HF,
                 stds_LF=self.stds_LF,
                 times=self.times,
                 times_LMST=self.times_LMST,
                 sol=self.sol,
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
            sol_end = float(solify(utct())) // 86400
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

    # def calc_noise_quantiles(self, qs,
    #                          sol_start=80, sol_end=None):
    #     if sol_end is None:
    #         # Now
    #         sol_end = float(solify(utct())) // 86400

    #     quantiles_HF = []
    #     quantiles_LF = []
    #     for q in qs:
    #         bol_LF = np.array([np.isfinite(self.stds_LF),
    #                            self.sol > sol_start,
    #                            self.sol < sol_end]).all(axis=0)
    #         quantiles_LF.append(
    #             np.quantile(a=20*np.log10(self.stds_LF[bol_LF]), q=q)
    #             )
    #         bol_HF = np.array([np.isfinite(self.stds_HF),
    #                            self.sol > sol_start,
    #                            self.sol < sol_end]).all(axis=0)
    #         quantiles_HF.append(
    #             np.quantile(a=20*np.log10(self.stds_HF[bol_HF]), q=q)
    #             )

    #     return quantiles_LF, quantiles_HF


    def plot_daystats(self,
                      cat: mqs_reports.catalog.Catalog = None,
                      sol_start: int = 80,
                      sol_end: int = 400):
        qs = [0.1, 0.25, 0.5, 0.9]
        if self.sols_quant is None:
            self.calc_quantiles(sol_end, sol_start, qs=qs)
            self.save_quantiles(fnam='noise_quantiles.npz')

        verts_LF = []
        verts_HF = []

        for i, isol in enumerate(self.sols_quant):
            if self.quantiles_LF[i, 0] > 0:
                verts_LF.append([isol, 10 * np.log10(self.quantiles_LF[i, 0])])
            if self.quantiles_HF[i, 0] > 0:
                verts_HF.append([isol, 10 * np.log10(self.quantiles_HF[i, 0])])
        verts_LF.append([verts_LF[-1][0], -300])
        verts_HF.append([verts_HF[-1][0], -300])
        verts_HF.append([self.sols_quant[0], -300])
        verts_LF.append([self.sols_quant[0], -300])

        fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex='all')
        ax_HF = ax[1]
        ax_LF = ax[0]

        poly = Polygon(verts_LF, facecolor='0.9', edgecolor='0.5')
        ax_LF.add_patch(poly)
        poly = Polygon(verts_HF, facecolor='0.9', edgecolor='0.5')
        ax_HF.add_patch(poly)

        rect = Rectangle(xy=(267, -300), width=21.5, height=200, zorder=10,
                         facecolor='darkgrey', edgecolor='black')
        ax_LF.add_patch(rect)
        rect = Rectangle(xy=(267, -300), width=21.5, height=200, zorder=10,
                         facecolor='darkgrey', edgecolor='black')
        ax_HF.add_patch(rect)
        cols = ['black', 'darkgrey', 'grey', 'darkgrey']
        ls = ['dashed', 'dashed', 'dashed', 'dashed']
        for i, q in enumerate(qs):
            ax_LF.plot(self.sols_quant, 10 * np.log10(self.quantiles_LF[:, i]),
                       label='%d%% of Sol' % (q * 100), c=cols[i], ls=ls[i])

        for i, q in enumerate(qs):
            ax_HF.plot(self.sols_quant, 10 * np.log10(self.quantiles_HF[:, i]),
                       label='%d%% of Sol' % (q * 100), c=cols[i], ls=ls[i])
        ax_HF.set_xlabel('Sol number')
        ax_LF.set_ylabel('PSD, displ. 2-6 sec. [dB]')
        ax_HF.set_ylabel('PSD, displ. 2-3 Hz [dB]')
        HF_times = []
        HF_amps = []
        HF_dists = []
        LF_times = []
        LF_amps = []
        LF_dists = []
        if cat is not None:
            cmap = plt.cm.get_cmap('plasma_r')
            cmap.set_over('royalblue')
            for event in cat.select(event_type=['HF', '24']):
                if event.distance is None:
                    HF_dists.append(50.)
                else:
                    HF_dists.append(event.distance)
                HF_times.append(solify(event.starttime).julday +
                                solify(event.starttime).hour / 60.)
                HF_amps.append(event.amplitudes['A_24'])

            for event in cat.select(event_type=['LF', 'BB']):
                if event.distance is None:
                    LF_dists.append(120.)
                else:
                    LF_dists.append(event.distance)
                amp_P = event.pick_amplitude(
                    pick='Peak_MbP',
                    comp='vertical',
                    fmin=1. / 6.,
                    fmax=1. / 1.5,
                    instrument='VBB'
                    )
                amp_S = event.pick_amplitude(
                    pick='Peak_MbS',
                    comp='vertical',
                    fmin=1. / 6.,
                    fmax=1. / 1.5,
                    instrument='VBB'
                    )
                amp = max(i for i in (amp_P, amp_S, 0.0) if i is not None)
                LF_times.append(solify(event.starttime).julday +
                                solify(event.starttime).hour / 60.)
                LF_amps.append(20 * np.log10(amp))

            sc = ax_LF.scatter(LF_times, LF_amps,
                               c=LF_dists, vmin=25., vmax=100., cmap=cmap,
                               edgecolors='k', linewidths=0.5,
                               s=80., marker='.', zorder=100)
            cax = plt.colorbar(sc, ax=ax_LF, use_gridspec=True)
            cax.ax.set_ylabel('distance / degree', rotation=270.,
                              labelpad=4.45)
            sc = ax_HF.scatter(HF_times, HF_amps,
                               c=HF_dists, vmin=5., vmax=30., cmap=cmap,
                               edgecolors='k', linewidths=0.5,
                               s=80., marker='.', zorder=100)
            cax = plt.colorbar(sc, ax=ax_HF, use_gridspec=True)
            cax.ax.set_ylabel('distance / degree', rotation=270.,
                              labelpad=12.45)

        sc = ax_HF.scatter(0, -300, label='Marsquake',
                           edgecolors='k', linewidths=0.5,
                           c='royalblue', s=80., marker='.')
        ax_LF.set_ylim(-210., -170.)
        ax_LF.set_title('Noise 2-8 seconds and LF/BB events')
        ax_HF.set_title('Noise 2-3 Hz and HF/2.4 Hz events')
        ax_HF.set_ylim(-225., -185.)
        ax_LF.set_xlim(sol_start, sol_end)
        for a in [ax_HF, ax_LF]:
            a.grid('on')
        plt.legend(loc='lower left')
        plt.tight_layout()
        # plt.savefig('noise_vs_eventamplitudes.pdf')
        plt.savefig('noise_vs_eventamplitudes.png', dpi=200)
        plt.show()

    def read_quantiles(self, fnam):
        data = np.load(fnam)
        self.quantiles_HF = data['quantiles_HF']
        self.quantiles_LF = data['quantiles_LF']
        self.sols_quant = data['sols']

    def save_quantiles(self, fnam):
        np.savez(file=fnam,
                 quantiles_HF=self.quantiles_HF,
                 quantiles_LF=self.quantiles_LF,
                 sols=self.sols_quant)

    def calc_quantiles(self, sol_end, sol_start,
                       qs):

        self.sols_quant = np.arange(sol_start, sol_end + 1)
        self.quantiles_LF = np.zeros((sol_end - sol_start + 1, len(qs)))
        self.quantiles_HF = np.zeros((sol_end - sol_start + 1, len(qs)))

        i = 0
        print('Calculating LF noise quantiles')
        for isol in tqdm(self.sols_quant):
            bol_sol = self.sol == isol - 1
            if sum(bol_sol) > 1:
                values = np.quantile(
                    a=np.ma.masked_less_equal(
                        x=self.stds_LF[bol_sol], value=0.0) ** 2. / (
                              1. / 1.5 - 1. / 6), q=qs)
                if np.isfinite(values).all():
                    self.quantiles_LF[i, :] = values
            i += 1

        i = 0
        print('Calculating HF noise quantiles')
        for isol in tqdm(self.sols_quant):
            bol_sol = self.sol == isol - 1
            if sum(bol_sol) > 1:
                values = np.quantile(
                    a=np.ma.masked_less_equal(x=self.stds_HF[bol_sol],
                                              value=0.0) ** 2. / 0.4, q=qs)
                if np.isfinite(values).all():
                    self.quantiles_HF[i, :] = values
            i += 1

        # Mask outliers
        self.quantiles_LF = np.ma.masked_less(self.quantiles_LF, value=1e-25)
        self.quantiles_HF = np.ma.masked_less(self.quantiles_HF, value=1e-25)
        self.quantiles_LF = np.ma.masked_greater(self.quantiles_LF, value=1e-8)
        self.quantiles_HF = np.ma.masked_greater(self.quantiles_HF, value=1e-8)


    def compare_events(self,
                       catalog = None,
                       threshold_dB: float = 3.):
        ratios = []
        for event in tqdm(catalog.select(event_type=['24', 'HF'])):
            nwins_below = sum(event.amplitudes['A_24'] >
                              threshold_dB + 20 * np.log10(self.stds_HF))
            event.ratio = nwins_below / len(self.stds_HF)
            ratios.append(event.ratio)

        for event in tqdm(catalog.select(event_type=['LF', 'BB'])):
            amp_P = event.pick_amplitude(
                pick='Peak_MbP',
                comp='vertical',
                fmin=1. / 6.,
                fmax=1. / 1.5,
                instrument='VBB'
                )
            amp_S = event.pick_amplitude(
                pick='Peak_MbS',
                comp='vertical',
                fmin=1. / 6.,
                fmax=1. / 1.5,
                instrument='VBB'
                )
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
    data = np.load(fnam)
    data_read = dict()
    for name in data.files:
        data_read[name] = data[name]

    return Noise(data=data_read)


if __name__ == '__main__':
    sc3_path = '/mnt/mnt_sc3data'
    inv = obspy.read_inventory('mqs_reports/data/inventory_VBB.xml')
    # noise = Noise(sc3dir=sc3_path,
    #               starttime=utct('20190202'),
    #               endtime=utct('20191026'),
    #               inv=inv,
    #               winlen_sec=120.
    #               )
    # noise.save('noise_0301_1025.npz')
    noise = read_noise('noise_0301_1025.npz')
    noise.plot_noise_stats()

    cat = Catalog(fnam_quakeml='mqs_reports/data/catalog_20191024.xml',
                  quality=['A', 'B', 'C', 'D'])

    cat.load_distances(fnam_csv='./mqs_reports/data/manual_distances.csv')
    cat.read_waveforms(inv=inv, sc3dir=sc3_path)
    cat.calc_spectra(winlen_sec=10.)
    # noise.compare_events(cat)
    noise.plot_daystats(cat)
