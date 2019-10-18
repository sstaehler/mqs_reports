from os.path import join as pjoin

import numpy as np
import obspy
from matplotlib import mlab as mlab
from obspy import UTCDateTime as utct
from obspy.signal.filter import envelope
from obspy.signal.rotate import rotate2zne
from obspy.signal.util import next_pow_2


def create_fnam_event(
        time,
        sc3dir,
        filenam_inst
        ):
    dirnam = pjoin(sc3dir, 'op/data/waveform/2019/XB/ELYSE/')
    dirnam_inst = pjoin(dirnam, '?H?.D')

    hour = utct(time).strftime('%H')
    fnam_inst = pjoin(dirnam_inst,
                      filenam_inst % utct(time).julday)
    if hour in ['00', '23']:
        fnam_inst = fnam_inst[:-1] + '?'

    return fnam_inst


def create_ZNE_HG(st: obspy.Stream,
                  inv: obspy.Inventory = None):
    # dip_u, dip_v, dip_w, = -35.3, -35.3, -35.3
    # azimuth_u, azimuth_v, azimuth_w = 0., 120., 240.

    if len(st) == 1 and st[0].stats.channel == 'EHU':
        # only SP1==SPZ switched on
        tr_Z = st[0].copy()
        tr_Z.stats.channel = 'EHZ'
        st_ZNE = obspy.Stream(traces=[tr_Z])

    else:
        chan_name = st[0].stats.channel[0:2]

        if inv is None:
            dip_u = -29.4
            dip_v = -29.2
            dip_w = -29.7
            azi_u = 135.1
            azi_v = 15.0
            azi_w = 255.0
        else:
            chan_u = inv.select(station='ELYSE',
                                starttime=st[0].stats.starttime,
                                endtime=st[0].stats.endtime,
                                channel=chan_name + 'U')[0][0][0]
            chan_v = inv.select(station='ELYSE',
                                starttime=st[0].stats.starttime,
                                endtime=st[0].stats.endtime,
                                channel=chan_name + 'V')[0][0][0]
            chan_w = inv.select(station='ELYSE',
                                starttime=st[0].stats.starttime,
                                endtime=st[0].stats.endtime,
                                channel=chan_name + 'W')[0][0][0]

            dip_u = chan_u.dip
            dip_v = chan_v.dip
            dip_w = chan_w.dip
            azi_u = chan_u.azimuth
            azi_v = chan_v.azimuth
            azi_w = chan_w.azimuth

        # st.resample(sampling_rate=100)
        for tr_1 in st:
            for tr_2 in st:
                tr_1.trim(starttime=tr_2.stats.starttime,
                          endtime=tr_2.stats.endtime)
        # st.decimate(5)

        st_ZNE = obspy.Stream()
        try:
            for tr_1 in st:
                for tr_2 in st:
                    # assert tr_1.stats.starttime == tr_2.stats.starttime
                    assert tr_1.stats.npts == tr_2.stats.npts

        except:
            print('Problem with rotating to ZNE:')
            print(st)
        else:
            if (len(st.select(channel=chan_name + 'U')) > 0 and
                    len(st.select(channel=chan_name + 'V')) > 0 and
                    len(st.select(channel=chan_name + 'W')) > 0):
                data_ZNE = \
                    rotate2zne(st.select(channel=chan_name + 'U')[0].data,
                               azi_u,
                               dip_u,
                               st.select(channel=chan_name + 'V')[0].data,
                               azi_v,
                               dip_v,
                               st.select(channel=chan_name + 'W')[0].data,
                               azi_w,
                               dip_w)
                for channel, data in zip(['Z', 'N', 'E'], data_ZNE):
                    tr = st.select(channel=chan_name + 'U')[0].copy()
                    tr.stats.channel = chan_name + channel
                    tr.data = data
                    st_ZNE += tr
    return st_ZNE


def read_data(fnam_complete, inv, kind, twin, fmin=1. / 20.):
    # if type(fnam_complete) is list:
    #     st = obspy.Stream()
    #     for f in fnam_complete:
    #         st += obspy.read(f,
    #                          starttime=twin[0] - 300.,
    #                          endtime=twin[1] + 300
    #                          )
    #     st.merge()
    # else:
    st = obspy.read(fnam_complete,
                    starttime=twin[0] - 300.,
                    endtime=twin[1] + 300
                    )
    st_seis = st.select(channel='?[LH]?')
    if len(st_seis) == 0:
        st_rot = obspy.Stream()
    else:
        st_seis.detrend(type='demean')
        st_seis.taper(0.1)
        st_seis.filter('highpass', zerophase=True, freq=fmin / 2.)
        st_seis.detrend()
        if st_seis[0].stats.starttime < utct('20190418T12:24'):
            correct_shift(st_seis.select(channel='??U')[0], nsamples=-1)
        for tr in st_seis:
            fmax = tr.stats.sampling_rate * 0.5
            tr.remove_response(inv,
                               pre_filt=(fmin / 2., fmin, fmax, fmax * 1.2),
                               output=kind)

        st_seis.merge(method=1)
        correct_subsample_shift(st_seis)

        st_rot = create_ZNE_HG(st_seis, inv=inv)
        if len(st_rot) > 0:
            if st_rot.select(channel='?HZ')[0].stats.channel == 'MHZ':
                fnam = fnam_complete[0:-32] + 'BZC' + fnam_complete[-29:-17] + \
                       '58.BZC' + fnam_complete[-11:]
                tr_Z = obspy.read(fnam,
                                  starttime=twin[0] - 900.,
                                  endtime=twin[1] + 900)[0]
                fmax = tr_Z.stats.sampling_rate * 0.45
                tr_Z.remove_response(inv,
                                     pre_filt=(0.005, 0.01, fmax, fmax * 1.2),
                                     output=kind)
                st_tmp = st_rot.copy()
                st_rot = obspy.Stream()
                tr_Z.stats.channel = 'MHZ'
                st_rot += tr_Z
                st_rot += st_tmp.select(channel='?HN')[0]
                st_rot += st_tmp.select(channel='?HE')[0]

            try:
                for tr in st_rot:
                    tr.data[np.isnan(tr.data)] = 0.
                st_rot.filter('highpass', zerophase=True, freq=fmin)
            except(NotImplementedError):
                # if there are gaps in the stream, return empty stream
                st_rot = obspy.Stream()
            else:
                st_rot.trim(starttime=twin[0], endtime=twin[1])

    return st_rot


def correct_subsample_shift(st):
    if len(st) > 1:
        shift = np.zeros(3)
        for i in range(1, 3):
            shift[i] = (st[i].stats.starttime - st[0].stats.starttime) % \
                       st[0].stats.delta
        if shift.sum() > 0:
            for i in range(1, 3):
                st[i].stats.starttime -= shift[i]


def correct_shift(tr, nsamples=-1):
    ltrace = tr.stats.npts
    if nsamples < 0:
        tr.data[0:ltrace + nsamples] = tr.data[-nsamples:ltrace]
    elif nsamples > 0:
        tr.data[nsamples:ltrace] = tr.data[0:ltrace - nsamples]
    return


def __dayplot_set_x_ticks(ax, starttime, endtime, sol=False):
    """
    Sets the xticks for the dayplot.
    """

    # day_break = endtime - float(endtime) % 86400
    # day_break -= float(day_break) % 1
    hour_ticks = []
    ticklabels = []
    interval = endtime - starttime
    interval_h = interval / 3600.
    ts = starttime
    tick_start = utct(ts.year, ts.month, ts.day, ts.hour)

    step = 86400
    if 0 < interval <= 60:
        step = 10
    elif 60 < interval <= 300:
        step = 30
    elif 300 < interval <= 900:
        step = 120
    elif 900 < interval <= 1800:
        step = 300
    elif 1800 < interval <= 7200:
        step = 600
    elif 7200 < interval <= 18000:
        step = 1800
    elif 18000 < interval <= 43200:
        step = 3600
    elif 43200 < interval <= 86400:
        step = 4 * 3600
    elif 86400 < interval:
        step = 6 * 3600
    step_h = step / 3600.

    # make sure the start time is a multiple of the step
    if tick_start.hour % step_h > 0:
        tick_start += 3600 * (step_h - tick_start.hour % step_h)

    # for ihour in np.arange(0, interval_h + step_h * 2, step_h):
    for ihour in np.arange(0, interval_h + 2 + step_h, step_h):
        hour_tick = tick_start + ihour * 3600.
        hour_ticks.append(hour_tick)
        if sol:
            ticklabels.append(utct(hour_tick).strftime('%H:%M:%S%nSol %j'))
        else:
            ticklabels.append(utct(hour_tick).strftime('%H:%M:%S%n%Y-%m-%d'))

    hour_ticks_minor = []
    for ihour in np.arange(0, interval_h, 1):
        hour_tick = tick_start + ihour * 3600.
        hour_ticks_minor.append(hour_tick)

    ax.set_xlim(float(starttime),
                float(endtime))
    ax.set_xticks(hour_ticks)
    ax.set_xticks(hour_ticks_minor, minor=True)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(float(starttime),
                float(endtime))


def calc_PSD(tr, winlen_sec):
    Fs = tr.stats.sampling_rate
    winlen = min(winlen_sec * Fs,
                 (tr.stats.endtime -
                  tr.stats.starttime) * Fs / 2.)
    NFFT = next_pow_2(winlen)
    pad_to = np.max((NFFT * 2, 1024))
    p, f = mlab.psd(tr.data,
                    Fs=Fs, NFFT=NFFT, detrend='linear',
                    pad_to=pad_to, noverlap=NFFT // 2)
    return f, p


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


def envelope_smooth(envelope_window, tr):
    tr_env = tr.copy()
    tr_env.data = envelope(tr_env.data)

    w = np.ones(int(envelope_window / tr.stats.delta))
    w /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, 'valid')

    return tr_env
