#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 2021

@author: Géraldine Zenhäusern
"""
from os import makedirs
from os.path import join as pjoin, exists as pexists

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import polarization.polarization as polarization
import seaborn as sns
from matplotlib.colorbar import make_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from obspy import Stream
from obspy import UTCDateTime as utct
from obspy.signal.util import next_pow_2
from scipy import stats
from mqs_reports.utils import detick



def plot_polarization_event_noise(waveforms_VBB, 
                              t_pick_P, t_pick_S, #Window in [sec, sec] around picks
                              timing_P, timing_S, timing_noise,#UTC timings for the three window anchors (start)
                              phase_P, phase_S, #Which phases/picks are used for the P and S windows: string with names
                              rotation = 'ZNE', BAZ=False,
                              BAZ_fixed=None, inc_fixed=None,
                              kind='spec', fmin=1., fmax=10.,
                              winlen_sec=20., overlap=0.5,
                              tstart=None, tend=None, vmin=None,
                              vmax=None, log=False, fname=None,
                              path='.',
                              dop_winlen=60, dop_specwidth=0.2,
                              nf=100, w0=20,
                              use_alpha=True, use_alpha2=False,
                              alpha_inc = 1.0, alpha_elli = 1.0, alpha_azi = 1.0,
                              f_band_density = (0.3, 1.), #frequency band for KDE curve analysis
                              plot_6C=False, plot_spec_azi_only = False, zoom = False,
                              differentiate=False, detick_1Hz=False,
                              impact = False, synthetics = False):

    """
    Plots polarisation of seismic event with window of noise and manually defined event time window
    """

    mod_180 = False #set to True if only mapping 0-180° azimuth, False maps 0-360°
    trim_time = [60., 300.] #[time before noise start, time after S] [seconds] Trims waveform
    
    st_Copy = waveforms_VBB.copy()  

    name_timewindows = [f'Signal {phase_P}', f'Signal {phase_S}', 'Noise', f'{phase_P}', f'{phase_S}'] #the last two are for the legend labeling


    #Rotate the waveforms into different coordinate system: ZRT or LQT
    if 'ZNE' not in rotation:
        if 'RT' in rotation:
            st_Copy.rotate('NE->RT', back_azimuth=BAZ)
            components = ['Z', 'R', 'T']
            #need to use -R, otherwise it aligns with 180° instead of 0°
            tr_R_data = st_Copy[1].data
            tr_R_data *= -1

        elif 'LQT' in rotation:
            st_Copy.rotate('ZNE->LQT', back_azimuth=BAZ, inclination = 40.0)
            components = ['L', 'Q', 'T']
        else:
            raise Exception("Sorry, please pick valid rotation system: ZNE, RT, LQT")
    else:
        components = ['Z', 'N', 'E']


    #differentiate waveforms
    if differentiate:
        st_Copy.differentiate()

    #trim the waveforms in length
    try:
        st_Copy.trim(starttime=utct(timing_noise[0]) - trim_time[0], #og: -50, +850
                     endtime=utct(timing_S) + trim_time[1])
    except ValueError: #if noise window is picked after the event
        st_Copy.trim(starttime=utct(timing_P) - trim_time[0]-120, #og: -50, +850
                     endtime=utct(timing_noise[-1]) + trim_time[0])


    st_Z = Stream(traces=[st_Copy.select(component=components[0])[0]])
    st_N = Stream(traces=[st_Copy.select(component=components[1])[0]])
    st_E = Stream(traces=[st_Copy.select(component=components[2])[0]])

    tstart_signal_P = utct(timing_P) + t_pick_P[0]
    tend_signal_P = utct(timing_S) - 20 if (utct(timing_P) + t_pick_P[1]) > (utct(timing_S) - 1) else  utct(timing_P) + t_pick_P[1]

    tstart_signal_S = utct(timing_S) + t_pick_S[0]
    tend_signal_S = utct(timing_S) + t_pick_S[1]


    #Noise window: MQS picks
    tstart_noise = utct(timing_noise[0]) # -120
    tend_noise = utct(timing_noise[-1])

    tstart, tend, dt = polarization._check_traces(st_Z, st_N, st_E, tstart, tend)

    #define in which row Signal and Noise hist are plotted
    signal_P_row = 2
    signal_S_row = 3
    noise_row = 1
    # density_row = 4

    # Create figure to plot in
    if plot_6C: #not tested
        gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],   # specgram, hist2d
                           height_ratios=[1, 1, 1, 1, 1, 1],
                           top=0.93,
                           bottom=0.05,
                           left=0.03,
                           right=0.96,
                           hspace=0.15,
                           wspace=0.08)
        box_legend = (1.3, 1.5)
        box_compass_colormap = [0.01, 0.01, 0.06] #offset left, top, width/height
        nrows = 6
        figsize_y = 12
    elif plot_spec_azi_only:
        gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],   # specgram, hist2d
                           height_ratios=[1, 1],
                           top=0.85,
                           bottom=0.08,
                           left=0.03,
                           right=0.98,
                           hspace=0.2,
                           wspace=0.08)
        box_legend = (1.3, 1.4)
        box_compass_colormap = [0.02, -0.02, 0.09] #offset left, top, width/height
        nrows = 2
        figsize_y = 6
    else:
        gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],  # specgram, hist2d, hist2d
                           height_ratios=[1, 1, 1, 1],
                           top=0.89,
                           bottom=0.05,
                           left=0.03, #0.05
                           right=0.98, #0.89
                           hspace=0.15,
                           wspace=0.08) #0.02 original
        box_legend = (1.2, 1.55)
        box_compass_colormap = [0.02, -0.02, 0.06] #offset left, top, width/height
        nrows = 4
        figsize_y = 9


    title = f'{fname}'


    # gridspec inside gridspec - nested subplots
    fig = plt.figure(figsize=(19, figsize_y))
    gs0 = gridspec.GridSpec(1, 2, figure=fig,
                            left=gridspec_kw['left'], bottom=gridspec_kw['bottom'], right=gridspec_kw['right'], top=gridspec_kw['top'],
                            wspace=0.1, hspace=None,
                            height_ratios=[1], width_ratios=[7, 1])

    #'Left' subplots
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows, 4, subplot_spec=gs0[0], wspace=gridspec_kw['wspace'], hspace=gridspec_kw['hspace'], height_ratios=gridspec_kw['height_ratios'], width_ratios=[4,1,1,1])
    axes0 = gs00.subplots()

    #'right' subplots - density curves
    # the following syntax does the same as the GridSpecFromSubplotSpec call above:
    gs01 = gs0[-1].subgridspec(nrows, 1, wspace=gridspec_kw['wspace'], hspace=gridspec_kw['hspace'], height_ratios=gridspec_kw['height_ratios'], width_ratios=[1])
    axes1 = gs01.subplots()



    if impact:
        title += f' - {rotation} impact: {impact}'
    if 'ZNE' not in rotation and not impact:
        title += f' - {rotation} rotated to {BAZ:.0f}° BAZ'

    #Mark the time window in the freq-time plot used for further analysis
    rect = [[None for i in range(3)] for j in range(nrows)] #prepare rectangles to mark the time windows
    color_windows = ['C0', 'Firebrick', 'grey', 'Peru'] #signal P, S, noise, density-color
    for j in range(nrows):
        rect[j][0] = patches.Rectangle((utct(tstart_signal_P).datetime,fmin+0.03*fmin),
                                       utct(tend_signal_P).datetime-utct(tstart_signal_P).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[0], fill = False) #signal
        rect[j][1] = patches.Rectangle((utct(tstart_signal_S).datetime,fmin+0.03*fmin),
                                       utct(tend_signal_S).datetime-utct(tstart_signal_S).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[1], fill = False) #signal
        rect[j][2] = patches.Rectangle((utct(tstart_noise).datetime,fmin+0.03*fmin),
                                       utct(tend_noise).datetime-utct(tstart_noise).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[2], fill = False) #noise



    winlen = int(winlen_sec / dt)
    nfft = next_pow_2(winlen) * 2

    # variables for statistics
    nbins = 30 #original: 90
    nts = 0

    # Calculate width of smoothing windows for degree of polarization analysis
    nfsum, ntsum, dsfacf, dsfact = polarization._calc_dop_windows(
        dop_specwidth, dop_winlen, dt, fmax, fmin,
        kind, nf, nfft, overlap, winlen_sec)

    if kind == 'spec':
        binned_data_signal_P = np.zeros((nrows, nfft // (2 * dsfacf) + 1, nbins))
        binned_data_signal_S = np.zeros_like(binned_data_signal_P)
        binned_data_noise = np.zeros_like(binned_data_signal_P)

        #for second plot with polar plots
        fBAZ_P = np.zeros((nfft // (2 * dsfacf) + 1, nbins))
        fBAZ_S = np.zeros_like(fBAZ_P)
        fBAZ_noise = np.zeros_like(fBAZ_P)

    else:
        binned_data_signal_P = np.zeros((nrows, nf // dsfacf, nbins))
        binned_data_signal_S = np.zeros_like(binned_data_signal_P)
        binned_data_noise = np.zeros_like(binned_data_signal_P)

        #for second plot with polar plots
        fBAZ_P = np.zeros((nf // dsfacf, nbins))
        fBAZ_S = np.zeros_like(fBAZ_P)
        fBAZ_noise = np.zeros_like(fBAZ_P)

    #For histogram curve
    kde_list = [[[] for j in range(3)] for _ in range(nrows)]
    kde_dataframe_P = [[] for _ in range(nrows)]
    kde_dataframe_S = [[] for _ in range(nrows)]
    kde_noiseframe = [[] for _ in range(nrows)]
    kde_weights = [[[] for j in range(3)] for i in range(nrows)]

    #custom colormap for azimuth
    color_list = ['blue', 'cornflowerblue', 'goldenrod', 'gold', 'yellow', 'darkgreen', 'green', 'mediumseagreen', 'darkred', 'firebrick', 'tomato', 'midnightblue', 'blue']
    custom_cmap =  LinearSegmentedColormap.from_list('', color_list) #interpolated colormap - or use with bounds
    bounds = [0, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 360]

    for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
        if tr_Z.stats.npts < winlen * 4:
            continue

        if detick_1Hz:
            tr_Z_detick = detick(tr_Z, 5)
            tr_N_detick = detick(tr_N, 5)
            tr_E_detick = detick(tr_E, 5)
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z_detick, tr_N_detick, tr_E_detick, kind, fmin, fmax,
                                         winlen, nfft, overlap, nf=nf, w0=w0)
        else:
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z, tr_N, tr_E, kind, fmin, fmax,
                                             winlen, nfft, overlap, nf=nf, w0=w0)

        azi1, azi2, elli, inc1, inc2, r1, r2, P = polarization.compute_polarization(
            u1, u2, u3, ntsum=ntsum, nfsum=nfsum, dsfacf=dsfacf, dsfact=dsfact)

        f = f[::dsfacf]
        t = t[::dsfact]
        t += float(tr_Z.stats.starttime)
        nts += len(t)
        #Prep bool mask for timing of the P, S, and noise window
        bol_signal_P_mask= np.array((t > tstart_signal_P, t< tend_signal_P)).all(axis=0)
        bol_signal_S_mask= np.array((t > tstart_signal_S, t< tend_signal_S)).all(axis=0)
        bol_noise_mask= np.array((t > tstart_noise, t< tend_noise)).all(axis=0)

        #get indexes where f lies in the defined f-band for density subplot
        twodmask_P = [[] for i in range(3)]
        twodmask_S = [[] for i in range(3)]
        twodmask_noise = [[] for i in range(3)]

        f_middle = f_band_density[0] + (f_band_density[1]-f_band_density[0])/2

        for i, (f_low, f_high) in enumerate(zip((f_band_density[0], f_band_density[0], f_middle),
                                                (f_band_density[1], f_middle, f_band_density[1]))):
            # Whole f-band, lower f-band, higher f-band
            bol_density_f_mask = np.array((f >= f_low, f < f_high)).all(axis=0)
            twodmask_P[i] = bol_density_f_mask[:, None] & bol_signal_P_mask[None, :]
            twodmask_S[i] = bol_density_f_mask[:, None] & bol_signal_S_mask[None, :]
            twodmask_noise[i] = bol_density_f_mask[:, None] & bol_noise_mask[None, :]


        #Scalogram and alpha/masking of signals
        # scalogram = 10 * np.log10((r1 ** 2).sum(axis=-1))
        # alpha, alpha2 = polarization._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2)
        if alpha_inc is not None:
            if alpha_inc > 0.: #S
                func_inc= np.cos
                func_azi= np.sin
                # func_azi= np.cos #S0173a special filtering for Cecilia
            else: #P
                alpha_inc= -alpha_inc
                func_inc= np.sin
                func_azi= np.cos
                # func_azi= np.sin #S0173a special filtering for Cecilia
        else:
            #look at azimuth without inclination, let's just set it like this.
            #So cosinus prefers P waves, set to sinus to prefer S waves (perpendicular to BAZ)
            func_azi= np.cos 

        r1_sum = (r1** 2).sum(axis=-1)
        if alpha_inc is not None:
            r1_sum *= func_inc(inc1)**(2*alpha_inc)
        if alpha_azi is not None:
            r1_sum *= abs(func_azi(azi1))**(2*alpha_azi)
        if alpha_elli is not None:
            r1_sum *= (1. - elli)**(2*alpha_elli)

        scalogram= 10 * np.log10(r1_sum)
        alpha, alpha2= polarization._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2)
        if mod_180:
            azi1= azi1 % np.pi
            azi2= azi2 % np.pi

        if alpha_inc is not None:
            alpha*= func_inc(inc1)**alpha_inc
        if alpha_azi is not None:
            alpha*= abs(func_azi(azi1))**alpha_azi
        if alpha_elli is not None:
            alpha*= (1. - elli)**alpha_elli


        #Prepare x axis array (datetime)
        t_datetime = np.zeros_like(t,dtype=object)
        for i, time in enumerate(t):
             t_datetime[i] = utct(time).datetime

        # plot scalogram, ellipticity, major axis azimuth and inclination

        iterables = [
            (scalogram, vmin, vmax, np.ones_like(alpha),
             'amplitude\n [dB]', np.arange(vmin, vmax+1, 20), 'plasma', None),
            (np.rad2deg(azi1), 0, 360, alpha,
             'major azimuth\n [degree]', np.arange(0, 361, 90), custom_cmap, bounds), #was 45 deg steps, tab20b
            (elli, 0, 1, alpha,
             'ellipticity\n', np.arange(0, 1.1, 0.2), 'gnuplot', None),
            (np.rad2deg(abs(inc1)), -0, 90, alpha,
             'major inclination\n [degree]', np.arange(0, 91, 20), 'gnuplot', None)]

        if plot_spec_azi_only:
            del iterables[-2:]
        if plot_6C:
            iterables.append(
                (np.rad2deg(azi2), 0, 180, alpha2,
                 'minor azimuth\n [degree]', np.arange(0, 181, 30), custom_cmap, bounds))
            iterables.append(
                (np.rad2deg(inc2), -90, 90, alpha2,
                 'minor inclination\n [degree]', np.arange(-90, 91, 30), 'gnuplot', None))

        for irow, [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] in \
                enumerate(iterables):

            ax = axes0[irow, 0]

            if log and kind == 'cwt':
                # imshow can't do the log sampling in frequency
                cm = polarization.pcolormesh_alpha(ax, t_datetime, f, data,
                                                   alpha=a, cmap=cmap,
                                                   vmin=rmin, vmax=rmax, bounds=boundaries)

            else:
                cm = polarization.imshow_alpha(ax, t_datetime, f, data, alpha=a, cmap=cmap,
                                  vmin=rmin, vmax=rmax)

            if tr_Z == st_Z[0]:
                cax, kw = make_axes(ax, location='left', fraction=0.07,
                                    pad=0.09) #pad=0.07
                plt.colorbar(cm, cax=cax, ticks=xticks, **kw)


            for i, mask in enumerate((twodmask_P[0], twodmask_S[0], twodmask_noise[0])):
                kde_list[irow][i] = data[mask]
                kde_weights[irow][i] = alpha[mask]
            for i in range(len(f)):
                binned_data_signal_P[irow, i, :] += np.histogram(data[i,bol_signal_P_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_signal_P_mask], density=True)[0]
                binned_data_signal_S[irow, i, :] += np.histogram(data[i,bol_signal_S_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_signal_S_mask], density=True)[0]
                binned_data_noise[irow, i, :] += np.histogram(data[i,bol_noise_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_noise_mask], density=True)[0]

    #set how many major and minor ticks for the time axis - concise date version
    loc_major = mdates.AutoDateLocator(tz=None, minticks=4, maxticks=15)
    loc_minor = mdates.AutoDateLocator(tz=None, minticks=4, maxticks=20)
    formatter = mdates.ConciseDateFormatter(loc_major)

    for ax in axes0:
        if zoom:
            ax[0].set_xlim(utct(utct(timing_P) - 120).datetime, utct(utct(timing_S) + 120).datetime)
        else:
            ax[0].set_xlim(utct(tstart).datetime, utct(tend).datetime)
        ax[0].xaxis.set_major_formatter(formatter)
        ax[0].xaxis.set_major_locator(loc_major)
        ax[0].xaxis.set_minor_locator(loc_minor)

        for a in ax[:]:
            a.set_ylim(fmin, fmax)
            a.set_ylabel("frequency [Hz]")
        if log:
            ax[0].set_yscale('log')
        ax[0].yaxis.set_ticks_position('both')
        ax[1].yaxis.set_ticks_position('both')
        ax[2].yaxis.set_ticks_position('both')
        # set tick position twice, otherwise labels appear right :/
        ax[signal_S_row].yaxis.set_ticks_position('right')
        ax[signal_S_row].yaxis.set_label_position('right')
        ax[signal_S_row].yaxis.set_ticks_position('both')

    for ax in axes1: #density
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('both')

    for ax in axes0[0:-1, :].flatten():
        ax.set_xlabel('')

    for ax in axes0[0:-1, 0]:
        ax.get_shared_x_axes().join(ax, axes0[-1, 0])

    for ax in axes0[:, 1]:
        ax.set_ylabel('')

    for ax in axes0[:, 2]:
        ax.set_ylabel('')
    for ax in axes0[:, 3]:
        ax.set_ylabel('frequency [Hz]', rotation=-90, labelpad=15)


    for i,ax in enumerate(axes0[:, 0]):
        ax.grid(b=True, which='major', axis='x')

        #Patched marking the hist time windows
        ax.add_patch(rect[i][0])
        ax.add_patch(rect[i][1])
        ax.add_patch(rect[i][-1])

        #mark P/S arrival
        ax.axvline(x=utct(timing_P).datetime,ls='dashed',c='black')
        ax.axvline(x=utct(timing_S).datetime,ls='dashed',c='black')


    for ax in axes0[0:-1, 0]:
        ax.set_xticklabels('')


    for i in range(nrows):
        kde_dataframe_P[i] = {'P': kde_list[i][0],
                            'weights': kde_weights[i][0]}
        kde_dataframe_S[i] = {'S': kde_list[i][1],
                            'weights': kde_weights[i][1]}
        kde_noiseframe[i] = {'Noise': kde_list[i][2],
                             'weights': kde_weights[i][2]}


    #Set titles, label the P and S timings, mark the boxes
    axes0[0, signal_P_row].set_title(f'{name_timewindows[0]} \n {t_pick_P[1]-t_pick_P[0]}s')
    axes0[0, signal_S_row].set_title(f'{name_timewindows[1]} \n {t_pick_S[1]-t_pick_S[0]}s')
    axes0[0, noise_row].set_title(f'{name_timewindows[2]} \n {tend_noise-tstart_noise:.0f}s')
    axes1[0].set_title(f'Density \n {f_band_density[0]}-{f_band_density[1]} Hz')
    axes0[0, 0].text(utct(tstart_signal_P).datetime, fmax+0.1*fmax, f'{name_timewindows[0]}', c=color_windows[0], fontsize=12)
    axes0[0, 0].text(utct(tstart_signal_S).datetime, fmax+0.1*fmax, f'{name_timewindows[1]}', c=color_windows[1], fontsize=12)
    if not zoom or (zoom and (utct(tstart_noise).datetime >= utct(utct(timing_P) - 120).datetime and \
                              utct(tstart_noise).datetime < utct(utct(timing_S) + 120).datetime)):
        axes0[0, 0].text(utct(tstart_noise).datetime, fmax+0.1*fmax, f'{name_timewindows[2]}', c=color_windows[2], fontsize=12)
    axes0[0, 0].text(utct(timing_P).datetime, fmin-0.35*fmin, phase_P, c='black', fontsize=12)
    axes0[0, 0].text(utct(timing_S).datetime, fmin-0.35*fmin, phase_S, c='black', fontsize=12)


    # linewidth_twofour = 1.0
    for irow, [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] in \
            enumerate(iterables):

        #hist plot: signal P
        ax = axes0[irow, signal_P_row]
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_signal_P[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')

        ax.set_ylim(fmin, fmax)
        ax.set_xticks(xticks)
        #Color the outside lines of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[0])
            spine.set_linewidth(2)

        #hist plot: signal S
        ax = axes0[irow, signal_S_row]
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_signal_S[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')

        ax.set_ylim(fmin, fmax)
        ax.set_xticks(xticks)
        #Color the outside lines of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[1])
            spine.set_linewidth(2)

        #hist plot: noise
        ax = axes0[irow, noise_row]
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_noise[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')


        ax.set_ylim(fmin, fmax)
        #Color the outside lines of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[2])
            spine.set_linewidth(2)


        if log:
            for i in range(0, 4):
                axes0[irow, i].set_yscale('log')
                axes0[irow, i].set_yticks((0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
                axes0[irow, i].set_yticklabels(("1/10", "1/5", "1/2", "1", "2", "5", "10"))
                axes0[irow, i].yaxis.set_minor_formatter(NullFormatter()) #removes minor ticks between the major ticks which are set above
                axes0[irow, i].set_ylim(fmin, fmax)
        ax.set_xticks(xticks)

        props = dict(boxstyle='round', facecolor='white', alpha=0.9)

        ax = axes0[irow, 0]
        ax.text(x=-0.25, y=0.5, transform=ax.transAxes, s=xlabel, #x=-0.18 #x=-019 gze
                ma='center', va='center', bbox=props, rotation=90, size=10)


       #density curves over some frequency band
        ax = axes1[irow]

        # sns.kdeplot(data=kde_dataframe[irow], ax=ax, common_norm=False, clip = (rmin, rmax), palette=[color_windows[0], color_windows[1]], legend=False, weights = kde_data_weights[irow], bw_adjust=.6)
        sns.kdeplot(data=kde_dataframe_P[irow], x='P', common_norm=False, ax=ax, clip = (rmin, rmax), color=color_windows[0], legend=False, weights = 'weights', bw_adjust=.6)
        sns.kdeplot(data=kde_dataframe_S[irow], x='S', common_norm=False, ax=ax, clip = (rmin, rmax), color=color_windows[1], legend=False, weights = 'weights', bw_adjust=.6)
        sns.kdeplot(data=kde_noiseframe[irow], x='Noise', common_norm=False, ax=ax, clip = (rmin, rmax), color=color_windows[2], fill=True, legend=False, weights = 'weights', bw_adjust=.6)


        ax.set_xticks(xticks)
        ax.set_xlim(rmin,rmax)
        ax.set_xlabel('')
        ax.set_yticklabels('')
        ax.set_yticks([])
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[3])
            spine.set_linewidth(2)


    #Get BAZ from max density of P curve, mark in density column
    max_x = [[],[]]
    for j, (i, xlim) in enumerate(zip((1,3), (360,90))):
        kernel = stats.gaussian_kde(kde_dataframe_P[i]['P'], weights = kde_dataframe_P[i]['weights'])
        kernel.covariance_factor = lambda : .17 #old:  lambda : .20
        kernel._compute_covariance()
        xs = np.linspace(-50,xlim+50,1000) #extend to positive and negative spaces so that the error can be wrapped around
        ys = kernel(xs)
        index = np.argmax(ys)
        max_x[j] = xs[index]
        
        #get the error of the BAZ from the full width of the half maximum
        #2021-07-23: may have issues if multiple peaks are present above halfway of maximum
        if j==0:
            #find the FWHM
            max_y = max(ys)
            indexes_ymax = [x for x in range(len(ys)) if ys[x] > max_y/2.0]
            left_edge = xs[min(indexes_ymax)]
            right_edge = xs[max(indexes_ymax)]
            
            # axes1[1].plot(xs, ys, color='yellow')
            # axes1[1].axvspan(left_edge, right_edge, color='blue', alpha=0.1) #marks the fwhm area, but not wrapping around zero
            # print(f'Error from {left_edge:.0f} to {right_edge:.0f}')
            if left_edge<0.:
                left_edge = 360.+left_edge #negative value, so 360-6, e.g
            if right_edge>360.:
                right_edge = right_edge-360.
        

    if BAZ_fixed and inc_fixed:
        BAZ_P = np.deg2rad(BAZ_fixed)
        inc_P = np.deg2rad(inc_fixed)
    else:
        BAZ_P = np.deg2rad(max_x[0])
        inc_P = np.deg2rad(max_x[1]) #needed later for polar plots

    title += ' - BAZ$_{KDE}^P$:'
    title += f' {max_x[0]:.0f}° - FWHM: {left_edge:.0f}°-{right_edge:.0f}°' #needs to be separated because f-strings and subscrips have issues
    ax = axes1[1]
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=max_x[0],c='r') #mark the polarisation BAZ from the maximum of the curve
    ax.scatter(max_x[0], ymax, color = 'r', marker = 'D', edgecolors = 'k', linewidths = 0.4, zorder = 100)

    #Set grid lines, mark BAZ
    if BAZ and ('ZNE' in rotation): #plot BAZ if it exists and if traces have NOT been rotated
        for ax in axes0[1, 1:]:
            ax.axvline(x=BAZ,ls='dashed',c='darkgrey')

        ax = axes1[1]
        ax.axvline(x=BAZ,ls='dashed',c='darkgrey')
        ax.scatter(BAZ, ymax, color = 'darkgrey', marker = 'v', edgecolors = 'k', linewidths = 0.4, zorder = 99)

    for ax in axes0[1:, 1:].flatten():
        ax.grid(b=True, which='both', axis='x', linewidth=0.2, color='grey')
    #Turn off y-axis ticks for left and middle histograms
    for ax in axes0[:, 1:-1].flatten():
        ax.set_yticklabels('')

    #Legend for density column
    colors = color_windows[:-1]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = [f'{name_timewindows[-2]}', f'{name_timewindows[-1]}', f'{name_timewindows[2]}']
    axes1[0].legend(lines, labels, loc='upper right', bbox_to_anchor=box_legend, fontsize=12, handlelength=0.8)


    #Compass rose-type plot to see in which direction azimuth colormap lies with respect to NESW
    rose_axes = fig.add_axes([gridspec_kw['left']-box_compass_colormap[0],
                              gridspec_kw['top']-box_compass_colormap[1],
                              box_compass_colormap[2], box_compass_colormap[2]], polar=True) # Left, Bottom, Width, Height
    if 'ZNE' not in rotation: #rotate the colormap so that 0° is in direction of the BAZ
        theta = [x+BAZ for x in bounds]
        theta = np.array(theta)
        theta[theta > 360] = theta[theta > 360] - 360 #remap values over 360°
    else:
        theta = bounds
    radii = [1]*len(theta)
    #Width of pie segments: first and last entry separately since blue is both at beginning and end of the color list= half the width each
    width = [30]*(len(theta)-2)
    width.insert(0, 15)
    width.insert((len(width)), 15)

    rose_axes.bar(np.radians(theta), radii, width=np.radians(width), color=color_list, align='edge')

    rose_axes.set_theta_zero_location("N")
    rose_axes.set_theta_direction('clockwise')
    rose_axes.set_xticks(np.radians(range(0, 360, 90)))
    rose_axes.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=8, )
    rose_axes.set_yticklabels('')
    rose_axes.tick_params(pad=-5.0)
    if BAZ:
        # rose_axes.axvline(x=np.radians(BAZ), color='black')
        rose_axes.annotate('', xytext=(0.0, 0.0), xy=(np.radians(BAZ),1.3),
                            arrowprops=dict(facecolor='grey', edgecolor='black', linewidth = 0.3, width=0.5, headwidth=4., headlength=4.),
                            xycoords='data', textcoords = 'data', annotation_clip=False)

        align_h = 'right' if BAZ > 180. else 'left'
        align_v = 'top' if 90. < BAZ < 270. else 'bottom'

        rose_axes.text(np.radians(BAZ), 1.3, 'BAZ', c='grey', fontsize=8, path_effects=[PathEffects.withStroke(linewidth=0.2, foreground="black")], horizontalalignment=align_h, verticalalignment = align_v)
    rose_axes.set_ylim([0, 1])

    fig.suptitle(title, fontsize=15)



    ## ----------------new figure for polar plots----------------
    #new figure with polar projections
    fig2, axes2 = plt.subplots(ncols=3, nrows=3, subplot_kw={'projection': 'polar'}, figsize=(10,11))
    fig2.subplots_adjust(hspace=0.4, wspace=0.3, top=0.87, bottom=0.05, left=0.13, right=0.93)

    colormap = 'gist_heat_r'

    BAZ_Inc_P = [[] for i in range(2)]
    BAZ_Inc_S = [[] for i in range(2)]
    BAZ_Inc_noise = [[] for i in range(2)]

    f_band_density_high = 3.
    bol_density_f_mask = np.array((f > f_band_density[0], f < f_band_density_high)).all(axis=0)

    [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] = iterables[1] #azimuth
    inc_data = np.rad2deg(abs(inc1)) #data inclination

    #Top row in plot: frequency vs BAZ
    for i in range(len(f)):
        fBAZ_P[i,:] += np.histogram(data[i,bol_signal_P_mask], bins=nbins, range=(rmin, rmax), weights=alpha[i,bol_signal_P_mask])[0]
        fBAZ_S[i,:] += np.histogram(data[i,bol_signal_S_mask], bins=nbins, range=(rmin, rmax), weights=alpha[i,bol_signal_S_mask])[0]
        fBAZ_noise[i,:] += np.histogram(data[i,bol_noise_mask], bins=nbins, range=(rmin, rmax), weights=alpha[i,bol_noise_mask])[0]

    #Following rows in plot: inclination vs BAZ
    for i in range(2):
        BAZ_Inc_P[i] = np.histogram2d(data[twodmask_P[i+1]], inc_data[twodmask_P[i+1]], bins=nbins, range=((rmin, rmax),(0,90)), weights=alpha[twodmask_P[i+1]])[0]
        BAZ_Inc_S[i] = np.histogram2d(data[twodmask_S[i+1]], inc_data[twodmask_S[i+1]], bins=nbins, range=((rmin, rmax),(0,90)), weights=alpha[twodmask_S[i+1]])[0]
        BAZ_Inc_noise[i] = np.histogram2d(data[twodmask_noise[i+1]], inc_data[twodmask_noise[i+1]], bins=nbins, range=((rmin, rmax),(0,90)), weights=alpha[twodmask_noise[i+1]])[0]


    #Vector fun
    #get P coordinates from kde curve maxima?
    BAZ_S = []
    inc_S = []

    #Define uP vector in cartesian coordinates from BAZ and inclination (inclination from polarisation is NOT the spherical coordinate inclination)
    gamma = np.linspace(0,2*np.pi, num=300)
    y = np.sin(np.pi/2-inc_P)*np.sin(BAZ_P)
    x = np.sin(np.pi/2-inc_P)*np.cos(BAZ_P)
    z = np.cos(np.pi/2-inc_P) #arbitrarily set r = 1
    uP = np.array([x, y, z])

    #get two orthogonal vectors uS1, uS2
    uS1 = np.random.randn(3)  # take a random vector
    uS1 -= uS1.dot(uP) * uP / np.linalg.norm(uP)**2       # make it orthogonal to uP
    uS1 /= np.linalg.norm(uS1)  # normalize it
    uS2 = np.cross(uP, uS1)      # cross product with uP to get second vector

    for i in gamma: #loop from 0 to 2pi
        uS = np.sin(i)*uS1 + np.cos(i)*uS2 #general vector uS from linear combination of uS1 and uS2
        r = np.sqrt(uS[0]**2+uS[1]**2+uS[2]**2)
        BAZ_S.append(np.arctan2(uS[1],uS[0]))
        inclination = np.pi/2-np.arccos(uS[2]/r) #inclination again defined as for polarisation analysis: 90° is vertical
        if inclination < 0: #'upper' part of sphere, ignore
            inc_S.append(np.nan)
        elif inclination <= np.pi/2:
            inc_S.append(inclination)
        else: #is landing on the other side, re-map to 0-90°
            inc_S.append(np.pi-inclination)


    #Plot all histograms
    P_hists = (fBAZ_P[bol_density_f_mask,:], BAZ_Inc_P[0].T, BAZ_Inc_P[1].T)
    S_hists = (fBAZ_S[bol_density_f_mask,:], BAZ_Inc_S[0].T, BAZ_Inc_S[1].T)
    Noise_hist = (fBAZ_noise[bol_density_f_mask,:], BAZ_Inc_noise[0].T, BAZ_Inc_noise[1].T)
    y_lim = (f[bol_density_f_mask], np.linspace(0, 90, nbins), np.linspace(0, 90, nbins))
    for i, (P, S, N, ylim) in enumerate(zip(P_hists, S_hists, Noise_hist, y_lim)):
        axes2[i,1].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, P,
                                cmap=colormap,
                                shading='auto')
        axes2[i,2].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, S,
                                cmap=colormap,
                                shading='auto')
        axes2[i,0].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, N,
                                cmap=colormap,
                                shading='auto')


    axes2[0,0].text(x=-0.45, y=0.5, transform=axes2[0,0].transAxes, s='major azimuth \n vs frequency', #x=-0.18 #x=-019 gze
                ma='center', va='center', bbox=props, rotation=90, size=15)
    axes2[1,0].text(x=-0.45, y=-0.2, transform=axes2[1,0].transAxes, s='major azimuth \n vs inclination', #x=-0.18 #x=-019 gze
                ma='center', va='center', bbox=props, rotation=90, size=15)

    for ax in axes2.flatten(): #all subplots
        ax.set_theta_zero_location("N")
        ax.set_theta_direction('clockwise')
        ax.grid(True)

    for i,ax in enumerate(axes2[0,:]): #top row
        ax.set_rlim(0)
        ax.set_rorigin(0.1)
        ax.yaxis.get_major_locator().base.set_params(nbins=4)
        ax.set_rscale('symlog')
        ax.set_rlim(f_band_density[0])
        # ax.set_rgrids((0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        ax.set_rgrids((0.5, 1., 1.5, 2., 2.5), labels=('0.5', '1.0', '1.5', '2.0', '2.5'))
        ax.set_title(f'{name_timewindows[i+2]} \n \n {f_band_density[0]}-{f_band_density_high} Hz', fontsize=15)

        if BAZ:
            ax.axvline(x=np.radians(BAZ), color='C0')
            ax.text(np.radians(BAZ), 3.5, 'BAZ', c='C0', fontsize=13)

    #lower rows
    for flat_ax, sub_title, rlim in zip((axes2[1,:].flatten(), axes2[2,:].flatten()),
                                  (f'{f_band_density[0]}-{f_middle:.2f} Hz', f'{f_middle:.2f}-{f_band_density[1]} Hz'),
                                  (-5, -5)):
        for ax in flat_ax:
            ax.set_title(sub_title, fontsize=15)
            ax.invert_yaxis()
            if BAZ:
                ax.axvline(x=np.radians(BAZ), color='C0')
                ax.text(np.radians(BAZ), rlim, 'BAZ', c='C0', fontsize=13)

    #Plot the orthogonal plane to the P wave
    for ax in axes2[1:,1:].flatten():
        ax.scatter(BAZ_P,np.rad2deg(inc_P), color='C9', zorder=100) #P-vector: point
        ax.plot(BAZ_S, np.rad2deg(inc_S), color= 'C9', zorder=101) #Orthogonal plane: line


    #Boxes to separate f-BAZ and inclination-BAZ plots
    rect = plt.Rectangle((0.01, 0.63), 0.98, 0.29, # (lower-left corner), width, height
                        fill=False, color="k", lw=2,
                        zorder=1000, transform=fig2.transFigure, figure=fig2)
    fig2.patches.extend([rect])

    rect2 = plt.Rectangle((0.01, 0.01), 0.98, 0.61, 
                          fill=False, color="k", lw=2,
                          zorder=1000, transform=fig2.transFigure, figure=fig2)
    fig2.patches.extend([rect2])

    fig2.suptitle(title, fontsize=18)


## ----------------save figures----------------

    if fname is None:
        plt.show()
    else:
        # savename = f'{fname}_diff' if differentiate else f'{fname}'
        savename = fname
        if plot_spec_azi_only:
            savename += '_2Row'
        elif plot_6C:
            savename += '_6Row'
        if zoom:
            savename += '_zoom'

        if impact:
            path_full = pjoin(path, f'Impact_search/Impact_{impact}')
        elif synthetics:
            path_full = pjoin(path, 'Synthetics')
        else:
            path_full = pjoin(path, 'Plots/Test')
            # path_full = pjoin(path)
            
        if not pexists(path_full):
            makedirs(path_full)    
            
        fig.savefig(pjoin(path_full, f'{savename}.png'), dpi=200)
        fig2.savefig(pjoin(path_full, f'{savename}_polarPlots.png'), dpi=200)
        

        # #save for automatic plots
        # fig.savefig(pjoin(path_full, f'{savename}_polarisation.png'), dpi=200)
        # if not zoom:
        #     fig2.savefig(pjoin(path_full, f'{savename}_polarPlots.png'), dpi=200)
        
    # np.savez(f'Data/{savename}_azimuth_P_filtered.npz', azimuth = np.rad2deg(azi1), alpha = alpha, cmap_colors = color_list, cmap_bounds = bounds, f = f, t = t_datetime) #for Cecilia

    plt.close('all')

