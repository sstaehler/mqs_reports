#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from argparse import ArgumentParser
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import obspy
from matplotlib import dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from obspy import UTCDateTime as utct
from obspy.taup import TauPyModel
# from pred_tstar.pred_tstar import get_dist
from taup_distance.taup_distance import get_dist

from mqs_reports.catalog import Catalog
from mqs_reports.utils import envelope_smooth


def define_arguments():
    helptext = 'Create Noise time evolution vs event amplitde overview plot'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('inventory', help=helptext)

    helptext = 'Path to SC3DIR'
    parser.add_argument('sc3_dir', help=helptext)
    return parser.parse_args()


def read_picks(fnam_csv):
    from csv import DictReader
    picks = dict()
    with open(fnam_csv, 'r') as csv_file:
        csv_reader = DictReader(csv_file)
        for row in csv_reader:
            picks[row['event_name'] + '_' + row['phase_name']] = row
    return picks


def main(args):
    inv = obspy.read_inventory(args.inventory)
    cat = Catalog(fnam_quakeml=args.input_quakeml,
                  type_select='lower', quality=['A', 'B', 'C'])
    # cat = cat.select(name=['S0235b'])
    # cat = cat.select(name=['S0409d'])

    cat.read_waveforms(inv=inv, sc3dir=args.sc3_dir)

    cat.calc_spectra(winlen_sec=30., detick_nfsamp=5)
    # detick_nfsamp=5)

    f_fit_lower = 0.15
    f_fit_upper = 0.8
    f_c = 0.7
    A0 = 8.5e-10
    waveform_list = ['S0235b', 'S0409d', 'S0484b', 'S0407a', 'S0173a']

    picks = read_picks('/opt/InSight_core/phases_events.csv')
    # model = TauPyModel(
    #         model="/opt/InSight_core/mean_model.npz")
    model = TauPyModel(
            model="/opt/InSight_core/model_core_downselection.npz")

    # model.get_travel_times(source_depth_in_km=50,
    #                        distance_in_degree=27,
    #                        phase_list=['ScS', 'ScP', 'S'])

    fig, ax = plt.subplots(ncols=2, nrows=1, sharey='all',
                           figsize=(11.69, 8.27))

    legends = {'P':         False,
               'S':         False,
               'ScS':       False,
               'ScP':       False,
               'ScS_pre':   False,
               'ScSScS':    False,
               'ScSScSScS': False,
               'SS':        False,
               'PP':        False,
               'PcP':       False}

    Pfac = {'P':         1.,  # 9./4.,
            'S':         1.,
            'ScS':       1.,
            'ScP':       1.,
            'ScS_pre':   1.,
            'ScSScS':    1.,
            'ScSScSScS': 1.,
            'SS':        1.,
            'PP':        1.,  # 9./4.,
            'PcP':       1.}  # , 9./4.}

    cols = {'P':         'C0',
            'S':         'C1',
            'ScS':       'C2',
            'ScP':       'C8',
            'ScS_pre':   'white',
            'ScSScS':    'C7',
            'ScSScSScS': 'C6',
            'SS':        'C3',
            'PP':        'C4',
            'PcP':       'C5'}

    fig_waveforms, ax_waveforms = plt.subplots(1, 1, figsize=(12, 8))

    amps = dict()
    amps_sig = dict()

    path_team_files = '/home/staehler/CloudStation/paper/2022_Attenuation/picks'
    path_output = '/home/staehler/CloudStation/paper/2020_Attenuation/figures'
    fnams_team = dict(
            CD='picks.csv',
            )
    team_picks = dict()
    for team, fnam in fnams_team.items():
        team_picks[team] = read_picks_teams(
                fnam_csv=pjoin(path_team_files, fnam))

    agreed_value = dict(
            S0173a=[512., 4.],
            # S0183a=[460., 20.],
            S0189a=[490., 30.],
            S0235b=[512., 3],
            S0407a=[510., 10.],
            S0409d=[512., 4.],
            S0484b=[513., 15],
            S0809a=[497., 10],
            S0784a=[515., 10],
            S0820a=[515., 10],
            S0325a=[499., 20.]
            )

    with PdfPages(pjoin(path_output, 'multipage_pdf.pdf')) as pdf:
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Spectral fits for marsquakes'
        d['Author'] = 'Simon Staehler'
        d['Subject'] = 'Spectral fits'
        d['Keywords'] = 'PdfPages multipage keywords author title subject'
        d['CreationDate'] = obspy.UTCDateTime().datetime
        for event_snickle, vals in picks.items():
            event_name = event_snickle[0:6]
            if event_name not in ['S0173a', 'S0325a', 'S0409d',
                                  'S0407a', 'S0189a', 'S0183a',
                                  'S0784a',
                                  'S0484b', 'S0235b'] or \
                    vals['phase_name'] not in ('P', 'S', 'ScS', 'SS', 'ScP'):
                continue
            if event_name not in amps:
                amps[event_name] = dict()
                amps_sig[event_name] = dict()
            if event_name not in ['S0167b', 'S0167b1']:
                ev = cat.select(name=event_name).events[0]
                if event_name in team_picks['A3'] and \
                        'P' in team_picks['A3'][event_name]:
                    print(event_name, ev.picks['P'],
                          team_picks['A3'][event_name]['P'])
                    ev.picks['P'] = str(team_picks['A3'][event_name]['P'])
                res = plot_single_phase(A0, ev,
                                        fname=event_snickle + '_fb.pdf',
                                        pdf=pdf,
                                        team_picks=team_picks,
                                        agreed_value=agreed_value,
                                        path_output=path_output,
                                        **vals)
                vals['tstar'] = - res[2] / np.pi
                vals['tstar_sig'] = res[3] / np.pi
                vals['A0'] = res[0] / np.log(10.) * 20.
                vals['A0_sig'] = res[1] / np.log(10.) * 20.

                amps[event_name][event_snickle[7:]] = vals['A0']
                amps_sig[event_name][event_snickle[7:]] = vals['A0_sig']

                ev.add_rotated_traces()
                print(event_snickle, vals['A0'], vals['A0_sig'])

                t_time = utct(ev.picks['P']) - ev.origin_time + float(
                        vals['tmin_amp'])
                # dist = ev.distance

                dist = get_dist(model=model,
                                tP=utct(ev.picks['P']),
                                tS=utct(ev.picks['S']),
                                phase_list=('P', 'S'),
                                depth=50.,
                                )[0]

                amplitude = float(vals['A0'])

                phase_name = vals['phase_name']
                if not legends[phase_name]:
                    label = phase_name
                    legends[phase_name] = True
                else:
                    label = None
                tstar = vals['tstar'] * Pfac[phase_name]
                ax[0].plot(t_time, tstar, 'o',
                           c=cols[phase_name],
                           label=label)
                ax[0].errorbar(x=t_time, y=tstar,
                               yerr=vals['tstar_sig'],
                               c=cols[phase_name])
                ax[1].plot(dist, tstar, 'o',
                           c=cols[phase_name],
                           label=label)
                ax[0].text(x=t_time, y=tstar, s=event_name, rotation=-45)
                ax[1].text(x=dist, y=0.1, s=event_name, rotation=-90)
                if vals['phase_name'][-3:] == 'ScS' and event_name in \
                        waveform_list:
                    y = None
                    if vals['comp'] == 'H':
                        comps = ['N', 'E']
                    else:
                        comps = [vals['comp']]
                    for comp in comps:
                        tr = ev.waveforms_VBB.select(channel='??' + comp)[
                            0].copy()
                        tr.filter('highpass', freq=1. / 2.4)
                        tr.filter('lowpass', freq=1. / 1.1)
                        tr.trim(starttime=utct(ev.picks['P']))
                        tr_env = envelope_smooth(tr=tr,
                                                 envelope_window_in_sec=6.0,
                                                 mode='same')
                        baseline = np.quantile(a=tr_env.data, q=0.1)
                        if y is None:
                            y = (tr_env.data - baseline) / \
                                np.quantile(a=abs(tr.data),
                                            q=0.985) * 10. + dist
                        else:
                            y += (tr_env.data - baseline) / \
                                 np.quantile(a=abs(tr.data), q=0.985) * 10.

                    x = tr.times() + float(
                        tr.stats.starttime - utct(ev.picks['P']))

                    ax_waveforms.fill_between(x, y1=y, y2=dist,
                                              alpha=0.4)
                    y = np.ma.masked_where(a=y,
                                           condition=y > np.quantile(a=y,
                                                                     q=0.985))
                    ax_waveforms.plot(x, y, label=event_snickle)
                    ax_waveforms.vlines(x=(float(vals['tmin_amp']),
                                           float(vals['tmax_amp'])),
                                        ymin=dist - 2., ymax=dist + 2.)

                # print(event_snickle, vals['phase_name'], float(vals[
                #       'tmin_amp']) + 5., tstar)

        plot_phase_amps(amps, amps_sig)

        x = np.arange(100, 300)
        ax[0].plot(x, x / 450., c=cols['P'], label='$Q_{eff}=450$')
        x = np.arange(350, 700)
        ax[0].plot(x, x / 300., c=cols['S'], label='$Q_{eff}=300$')
        x = np.arange(500, 800)
        ax[0].plot(x, x / 500., c=cols['ScS'], label='$Q_{eff}=500$')

        ax[0].set_ylim(0, 2.5)
        ax[1].set_xlim(0, 75)
        ax[0].set_xlim(0, 1000.)
        ax[0].set_ylabel('t* / seconds')
        ax[0].set_xlabel('travel time / seconds')
        ax[1].set_xlabel('distance / degree')
        ax[0].legend()
        ax[0].legend()

        pdf.savefig()
        ax_waveforms.set_xlim(450, 600)
        ax_waveforms.set_xlabel('time after P-arrival / seconds')
        plot_expected_times(ax=ax_waveforms,
                            model=model,
                            phase=('SS', 'ScSScS'),
                            depth=50.)
        plot_expected_times(ax=ax_waveforms,
                            model=model,
                            phase=('S', 'ScSScS'),
                            depth=50.)
        plot_expected_times(ax=ax_waveforms,
                            model=model,
                            phase=('ScS', 'ScSScS'),
                            depth=50.)
        ax_waveforms.legend()
    plt.show()
    return


def plot_phase_amps(amps, amps_sig):
    fig_amps, ax_amps = plt.subplots(1, 1, figsize=(3.5, 6))
    ievent = 0
    for event, amplitude in amps.items():
        # if 'ScS' in amplitude and 'S' in amplitude:
        if event in ('S0235b', 'S0407a', 'S0409d', 'S0325a',
                     'S0189a', 'S0484b',
                     'S0173a'):
            ievent += 1
            if 'SS' in amplitude:
                ax_amps.plot((0, 1, 2),
                             (amplitude['S'],
                              amplitude['SS'],
                              amplitude['ScS']),
                             c='C%d' % ievent,
                             label=event)
                ax_amps.errorbar(x=(0, 1, 2),
                                 y=(amplitude['S'],
                                    amplitude['SS'],
                                    amplitude['ScS']),
                                 yerr=(amps_sig[event]['S'],
                                       amps_sig[event]['SS'],
                                       amps_sig[event]['ScS']),
                                 c='C%d' % ievent,
                                 marker='o'
                                 )
            else:
                ax_amps.plot((0, 2),
                             (amplitude['S'],
                              amplitude['ScS']),
                             c='C%d' % ievent,
                             label=event)
                ax_amps.errorbar(x=(0, 2),
                                 y=(amplitude['S'],
                                    amplitude['ScS']),
                                 yerr=(amps_sig[event]['S'],
                                       amps_sig[event]['ScS']),
                                 c='C%d' % ievent,
                                 marker='o'
                                 )
    ax_amps.set_xticks((0, 1, 2))
    ax_amps.set_xticklabels(('S', 'SS', 'ScS'))
    ax_amps.set_xlabel('phase name')
    ax_amps.set_ylabel('zero-frequency power / ($m^2/Hz$) [dB]')
    ax_amps.legend()
    fig_amps.tight_layout()
    fig_amps.savefig('fig_2_core.pdf')
    plt.show()


def plot_expected_times(ax: plt.Axes,
                        model: TauPyModel,
                        phase=('S'),
                        depth=50.):
    distances = np.arange(5, 80, 5)
    times = []
    dists = []
    for dist in distances:
        arrival_P = model.get_travel_times(
                distance_in_degree=dist,
                source_depth_in_km=depth,
                phase_list='P')[0]
        if phase == 'S2':
            arrivals = model.get_travel_times(
                    distance_in_degree=dist,
                    source_depth_in_km=depth,
                    phase_list='S')
            if len(arrivals) > 1:
                dists.append(dist)
                times.append(arrivals[1].time - arrival_P.time)

        else:
            arrivals = model.get_travel_times(
                    distance_in_degree=dist,
                    source_depth_in_km=depth,
                    phase_list=phase)
            if len(arrivals) > 0:
                dists.append(dist)
                times.append(arrivals[0].time - arrival_P.time)

    ax.plot(times, dists, label=phase[0])


def calc_mt_spec(tr, t_ref, tmin_amp, tmax_amp,
                 return_vel=False):
    from mqs_reports.utils import detick
    import mtspec

    tr_detick = detick(tr, detick_nfsamp=5)

    tr_amp = tr_detick.slice(starttime=t_ref + tmin_amp,
                             endtime=t_ref + tmax_amp)
    res = mtspec.mtspec(data=tr_amp.data,
                        delta=tr_amp.stats.delta,
                        time_bandwidth=2.5,
                        statistics=True
                        )
    f = res[1]
    p = np.sqrt(res[0])
    p_low = np.sqrt(res[2][:, 0])
    p_up = np.sqrt(res[2][:, 1])
    omega = 2 * np.pi * f
    if return_vel:
        return f, p * omega, p_low * omega, p_up * omega
    else:
        return f, p, p_low, p_up


def plot_single_phase(A0, ev, event_name, phase_name,
                      comp, f_c, f_fit_lower,
                      f_fit_upper,
                      tmax_amp, tmin_amp, fname=None,
                      team_picks=None,
                      agreed_value=None,
                      path_output='.',
                      pdf=None):
    if event_name == 'S0167b':
        ev.origin_time = utct('2019-05-17T19:23:07')
        ev.picks['P'] = '2019-05-17T19:31:38.091488Z'
        ev.picks['S'] = utct(ev.picks['P']) + 418.
        ev.distance = 70.
    ev.baz = 70.
    ev.add_rotated_traces()
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(11.69, 8.27))
    if event_name in agreed_value:
        pick_all = agreed_value[event_name]
    else:
        pick_all = [0., 0.]

    if phase_name == 'ScS':
        tmin_amp = pick_all[0] - \
                   pick_all[1] - 10.
        tmax_amp = pick_all[0] + \
                   pick_all[1] + 10.
    else:
        tmin_amp = float(tmin_amp)
        tmax_amp = float(tmax_amp)
    f_fit_lower = float(f_fit_lower)
    f_fit_upper = float(f_fit_upper)
    f_c = float(f_c)

    if tmin_amp < 1000:
        t_ref = utct(ev.picks['P'])
    else:
        t_ref = 0.

    if comp == 'H':
        comp_fb = 'N'
    else:
        comp_fb = comp
    f, p = ev.plot_filterbank_phase(comp=comp_fb,
                                    starttime=ev.starttime - 100,
                                    endtime=ev.starttime + 1000.,
                                    fmin=1. / 16.,
                                    fmax=4.,
                                    df=2 ** (0.25),
                                    waveforms=False,
                                    ax_fbs=ax[0],
                                    zerophase=True,
                                    fmin_mark=f_fit_lower,
                                    fmax_mark=f_fit_upper,
                                    tmin_plot=tmin_amp - 25.,
                                    tmax_plot=tmax_amp + 25.,
                                    tmin_amp=tmin_amp,
                                    tmax_amp=tmax_amp)
    # tmin_amp=410.,
    # tmax_amp=450.)

    # if float(tmin_amp) < 1000:

    if comp == 'H':
        f_mt, p_mt_N, p_low_mt_N, p_up_mt_N = calc_mt_spec(
                tr=ev.waveforms_VBB.select(channel='??N')[0],
                t_ref=t_ref,
                tmin_amp=tmin_amp, tmax_amp=tmax_amp)
        f_mt, p_mt_E, p_low_mt_E, p_up_mt_E = calc_mt_spec(
                tr=ev.waveforms_VBB.select(channel='??E')[0],
                t_ref=t_ref,
                tmin_amp=tmin_amp, tmax_amp=tmax_amp)
        p_mt = (p_mt_E + p_mt_N) / 2.
        p_low_mt = (p_low_mt_E + p_low_mt_N) / 2.
        p_up_mt = (p_up_mt_E + p_up_mt_N) / 2.
        # ax[1].plot(f_mt, p_mt_N, c='k', ls=':', lw=1.0)
        # ax[1].plot(f_mt, p_mt_E, c='k', ls=':', lw=1.0)

    else:
        f_mt, p_mt, p_low_mt, p_up_mt = calc_mt_spec(
                tr=ev.waveforms_VBB.select(channel='??' + comp)[0],
                t_ref=t_ref,
                tmin_amp=tmin_amp, tmax_amp=tmax_amp)
    # ax[1].plot(f_mt, p_mt, c='k', lw=1.5, label='signal amplitude')
    ax[1].errorbar(x=f_mt, y=p_mt,
                   yerr=np.asarray([p_mt - p_low_mt, p_up_mt - p_mt]),
                   ecolor='dimgrey', capsize=1.5, fmt='o', ms=2.,
                   mfc='darkgrey', mec='black',
                   lw=1.0, label='signal amplitude')
    # ax[1].fill_between(x=f_mt, y1=p_low_mt, y2=p_up_mt, alpha=0.3,
    #                    facecolor='grey', label='amplitude uncertainty')

    ax[1].plot(ev.spectra['noise']['f'],
               np.sqrt(ev.spectra['noise']['p_H']) *
               1,  # (2 * np.pi * ev.spectra['noise']['f']),
               c='k', ls='dotted', label='noise amplitude')
    # ax[1].plot(ev.spectra['S']['f'],
    #            np.sqrt(ev.spectra['S']['p_H']),
    #            c='grey', ls='dotted')
    # ax[1].plot(f, p, label='signal, phase')

    # for i in range(len(f)):
    #    ax[1].plot(f[i], p[i], c='C%d' % (i % 10), marker='o')

    # ax[1].plot(f_mt, p_mt * (1 + (f_mt / f_c) ** 2), c='k', ls=':', lw=1.5)

    fit_bol = np.array((f_mt > float(f_fit_lower),
                        f_mt < float(f_fit_upper))).all(axis=0)
    # res_fit = np.polyfit(x=f_mt[fit_bol],
    #                      y=np.log(p_mt[fit_bol] *
    #                               (1 + (f_mt[fit_bol] / f_c) ** 2)
    #                               ),
    #                      deg=1)
    from mqs_reports.utils import linregression, spectral_fit
    res_fit = linregression(x=f_mt[fit_bol],
                            y=np.log(p_mt[fit_bol] *
                                     (1 + (f_mt[fit_bol] / f_c) ** 2)))

    f_noise = ev.spectra['noise']['f']
    # fit_bol2 = np.array((f_noise > float(f_fit_lower),
    #                      f_noise < float(f_fit_upper))).all(axis=0)
    fit_bol2 = np.array((f_noise > float(0.05),
                         f_noise < float(2.0))).all(axis=0)
    p_mt_ipl = np.interp(x=f_noise[fit_bol2], xp=f_mt,
                         fp=p_mt)
    p_mt_sigma = (p_up_mt - p_low_mt) / 2.
    p_mt_sigma_ipl = np.interp(x=f_noise[fit_bol2], xp=f_mt, fp=p_mt_sigma)
    omega = 2. * np.pi * f_noise[fit_bol2]
    spectral_fit(f=f_noise[fit_bol2],
                 p_signal=p_mt_ipl * omega,
                 sigma_signal=p_mt_sigma_ipl * omega,
                 p_noise=np.sqrt(ev.spectra['noise']['p_Z'][fit_bol2]) * omega,
                 fnam='/tmp/fit_event_%s_%s.png' % (event_name, phase_name),
                 fmin=-1, fmax=-1)

    fnam = pjoin(path_output, 'mt_spec_%s_%s.npz' % (event_name, phase_name))
    np.savez(fnam, res_fit=res_fit, f_mt=f_mt, p_mt=p_mt,
             p_low_mt=p_low_mt, p_up_mt=p_up_mt)

    ax[1].plot(f_mt, np.exp(res_fit[0]) * np.exp(res_fit[2] * f_mt),
               label='fit (attenuation only)',
               lw=1.5, ls='dashed', c='darkgreen')
    # label='fit t*=%4.2f' % (res_fit[2] / np.pi))
    ax[1].plot(f_mt, np.exp(res_fit[0]) * np.exp(res_fit[2] * f_mt) /
               (1 + (f_mt / f_c) ** 2), label='fit (with source)',
               lw=1.5, c='darkgreen')

    ax_utc = ax[0].twiny()
    ax_utc.set_xlim((utct(ev.picks['P']) + (tmin_amp - 25.)).datetime,
                    (utct(ev.picks['P']) + (tmax_amp + 25.)).datetime
                    )
    if team_picks is not None:
        iteam = 0
        for team, team_pick in team_picks.items():
            for event, picks in team_pick.items():
                if event == event_name:
                    for phase, time in picks.items():
                        ax_utc.axvline(x=time, lw=2, color='C%d' % iteam,
                                       )
                        ax_utc.annotate(xy=(time, 21. + iteam * 0.8),
                                        color='white',
                                        horizontalalignment='center',
                                        bbox=dict(boxstyle='square', pad=0.1,
                                                  fc='C%d' % iteam),
                                        text='%s' % (team))
                        # ax_utc.text(x=time, y=21. + iteam * 0.8,
                        #             color='white',
                        #             horizontalalignment='center',
                        #             bbox=dict(boxstyle='square', pad=0.1,
                        #                       fc='C%d' % iteam),
                        #             s='%s' % (team))
                        ax_utc.errorbar(x=time, y=21. + iteam * 0.8,
                                        xerr=0)
            iteam += 1

    if agreed_value is not None:
        if event_name in agreed_value:
            ax_utc.axvline(x=utct(ev.picks['P']) + pick_all[0], color='black',
                           lw=3., ls='dashed')
            ax_utc.errorbar(x=utct(ev.picks['P']) + pick_all[0],
                            y=-0.1, capsize=3., zorder=5., lw=3.,
                            xerr=pick_all[1], c='black')  # , zorder=210)
            ax[0].fill_between(x=(pick_all[0] - pick_all[1],
                                  pick_all[0] + pick_all[1]),
                               y1=-10 * np.ones(2), y2=40 * np.ones(2),
                               color='lightgrey')
            anno = ax_utc.annotate(xy=(utct(ev.picks['P']) + pick_all[0], -1),
                                   color='black',
                                   horizontalalignment='center',
                                   bbox=dict(boxstyle='square', pad=0.4,
                                             fc='white'),
                                   text='agreed pick',
                                   )
            anno.set_zorder(10)
    locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator,
                                            offset_formats=['', '%Y', '%Y-%b',
                                                            '%Y-%b-%d',
                                                            '%Y-%b-%d',
                                                            '%Y-%b-%d %H:%M'],
                                            zero_formats=['', '%Y', '%b',
                                                          '%b-%d', '%H:%M',
                                                          '%H:%M:%S'])
    ax_utc.xaxis.set_major_locator(locator)
    ax_utc.xaxis.set_major_formatter(formatter)
    ax_utc.set_xlabel('time / UTC')

    ax[1].set_ylim(6e-12, 1e-8)
    ax[1].fill_between(x=(0, f_fit_lower), y1=1e-13, y2=1e-6,
                       color='orangered', alpha=0.1, zorder=-1)
    ax[1].fill_between(x=(f_fit_upper, 2.), y1=1e-13, y2=1e-6,
                       color='orangered', alpha=0.1, zorder=-1,
                       label='signal below noise')
    ax[1].set_xlim(0.025, 1.52)
    # ax[1].set_xscale('log')
    # ax[1].set_xticks((1./16, 1./8., 1./ 4., 1./2., 1., 2.))
    # ax[1].set_xticklabels(("1/16", "1/8", "1/4", "1/2", "1", "2"))
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel('ASD / $\mathrm{m}/\sqrt{\mathrm{Hz}}$')
    ax[1].set_xlabel('frequency / Hz')
    fig.suptitle('%s %s %s' % (event_name, phase_name, comp))
    if fname is None:
        plt.show()
    if fname is not None:
        fig.savefig(pjoin(path_output, fname), dpi=200)
    if pdf is not None:
        pdf.savefig()
    plt.close(fig)

    return res_fit


def read_picks_teams(fnam_csv):
    from csv import DictReader
    picks = dict()
    event_info = dict()
    with open(fnam_csv, 'r') as csv_file:
        csv_reader = DictReader(csv_file)
        for row in csv_reader:
            # print(row)
            if row['tS'] == '':
                tref = utct(row['tP'])
                if row['tScS_tP'] is not '':
                    picks[row['event']] = dict(
                            S=tref + float(row['tS_tP']),
                            P=tref,
                            ScS=tref + float(row['tScS_tP']),
                            )

            else:
                picks[row['event']] = dict(
                        S=utct(row['tS']),
                        P=utct(row['tP']),
                        ScS=utct(row['tScS']),
                        )
    return picks


if __name__ == '__main__':
    args = define_arguments()
    main(args)
