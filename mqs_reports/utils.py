import glob
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import obspy
from matplotlib import mlab as mlab
from obspy import UTCDateTime as utct
from obspy.signal.filter import envelope
from obspy.signal.rotate import rotate2zne
from obspy.signal.util import next_pow_2
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert

SEC_PER_DAY_EARTH = 86400
SEC_PER_DAY_MARS = 88775.2440


def solify(UTC_time, sol0=UTCDateTime(2018, 11, 26, 5, 10, 50.33508)):
    if type(UTC_time) is str:
        UTC_time = UTCDateTime(UTC_time)
    MIT = (UTC_time - sol0) / SEC_PER_DAY_MARS
    t = UTCDateTime((MIT - 1) * SEC_PER_DAY_EARTH)
    return t


def UTCify(LMST_time, sol0=UTCDateTime(2018, 11, 26, 5, 10, 50.33508)):
    MIT = float(LMST_time) / SEC_PER_DAY_EARTH + 1
    UTC_time = UTCDateTime(MIT * SEC_PER_DAY_MARS + float(sol0))
    return UTC_time


def create_fnam_event(
        time,
        sc3dir,
        filenam_inst
        ):
    dirnam = pjoin(sc3dir,
                   'op/data/waveform/%04d/XB/ELYSE/' % utct(time).year)
    dirnam_inst = pjoin(dirnam, '???.D')

    hour = utct(time).strftime('%H')
    fnam_inst = pjoin(dirnam_inst,
                      filenam_inst % (utct(time).year, utct(time).julday))
    if hour in ['00', '23']:
        fnam_inst = fnam_inst[:-1] + '?'

    return fnam_inst


def f_c(M0, vs, ds):
    # Calculate corner frequency for event with M0,
    # assuming a stress drop ds
    return 4.9e-1 * vs * (ds / M0) ** (1 / 3)


def M0(Mw):
    return 10 ** (Mw * 1.5 + 9.1)

def attenuation_term(freqs, Qm, Qk=5e4, x=1e6, phase='S', vp=7.5e3, vs=4.1e3):
    if phase == 'P':
        L = 4 / 3 * (vs / vp) ** 2
        Q = 1 / (L / Qm + (1 - L) / Qk)
    else:
        Q = Qm
    return np.exp(-np.pi * x * freqs / vs / Q)


def pred_spec(freqs, ds, Qm, amp, dist, mag, phase, vs=5.e3):
    stf_amp = 1 / (1 + (freqs / f_c(M0=M0(mag),
                                    vs=vs, ds=ds)
                        ) ** 2)
    A = attenuation_term(freqs, Qm=Qm, x=dist, vs=vs, phase=phase)
    return 20 * np.log10(A * stf_amp) + amp


# def attenuation_term(freqs, Qm, Qk=5e4, x=1e6, phase='S', exp=0.0, f0=1,
#                      vp=7.5e3, vs=4.1e3):
#     if phase == 'P':
#         vel = vp
#         L = 4 / 3 * (vs / vp) ** 2
#         Q = 1 / (L / Qm + (1 - L) / Qk)
#     else:
#         vel = vs
#         Q = Qm
#     Q = Q * (freqs / f0) ** exp
#     Qscat = 300
#     Q = 1. / (1. / Q + 1 / Qscat)
#     return np.exp(-np.pi * x / vel * freqs / Q)
#
#
# def pred_spec(freqs, ds, Qm, amp, dist, mag, phase='S', vp=7.5e3, vs=4.2e3):
#     stf_amp = 1 / (1 + (freqs / f_c(M0=M0(mag),
#                                     vs=2.8e3, ds=ds)
#                         )                    filenam_pressure = 'XB.ELYSE.02.MDO.D.%d.%03d')
#     A = attenuation_term(freqs, Qm=Qm, x=dist, phase=phase, vp=vp, vs=vs)
#     return 20 * np.log10(A * stf_amp) + amp


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

        for tr_1 in st:
            for tr_2 in st:
                tr_1.trim(starttime=tr_2.stats.starttime,
                          endtime=tr_2.stats.endtime,
                          nearest_sample=True)
        st_ZNE = obspy.Stream()
        try:
            for tr_1 in st:
                for tr_2 in st:
                    # assert tr_1.stats.starttime == tr_2.stats.starttime
                    #assert tr_1.stats.npts == tr_2.stats.npts
                    if not tr_1.stats.npts == tr_2.stats.npts:
                        tr_1.data = tr_1.data[0:tr_2.stats.npts]

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
    if len(glob.glob(fnam_complete)) > 0:
        st = obspy.read(fnam_complete,
                        starttime=twin[0] - 300.,
                        endtime=twin[1] + 300,
                        nearest_sample=False
                        )
        # st = obspy.read(fnam_complete)
        # st.trim(starttime=twin[0] - 300.,
        #         endtime=twin[1] + 300)
        st_seis = st.select(channel='?[LH]?')
        st_seis.merge(method=1, fill_value='interpolate')
        st_seis.detrend(type='demean')
        st_seis.taper(0.1)
        st_seis.filter('highpass', zerophase=True, freq=fmin / 2.)
        st_seis.detrend()
        # correct_subsample_shift(st_seis)
        if len(st_seis) > 0:
            if st_seis[0].stats.starttime < utct('20190418T12:24'):
                correct_shift(st_seis.select(channel='??U')[0], nsamples=-1)
            for tr in st_seis:
                fmax = tr.stats.sampling_rate * 0.5
                pre_filt = (fmin / 2., fmin, fmax*1.2, fmax * 1.5)
                remove_response_stable(tr, inv, output=kind,
                                       pre_filt=pre_filt)

            st_rot = create_ZNE_HG(st_seis, inv=inv)
            if len(st_rot) > 0:
                if st_rot.select(channel='??Z')[0].stats.channel == 'MHZ':
                    fnam = fnam_complete[0:-32] + 'BZC' + \
                           fnam_complete[-29:-17] + \
                           '58.BZC' + fnam_complete[-11:]
                    tr_Z = obspy.read(fnam,
                                      starttime=twin[0] - 900.,
                                      endtime=twin[1] + 900)[0]
                    fmax = tr_Z.stats.sampling_rate * 0.45
                    tr_Z.remove_response(inv,
                                         pre_filt=(
                                             0.005, 0.01, fmax, fmax * 1.2),
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
        else:
            st_rot = obspy.Stream()
    else:
        st_rot = obspy.Stream()
    return st_rot


def remove_response_stable(tr, inv, **kwargs):
    try:
        tr.remove_response(inv, **kwargs)
    except ValueError:
        filtered_inv = inv.select(
            location=tr.stats.location, channel=tr.stats.channel,
            starttime=tr.stats.starttime - 7 * 86400,
            endtime=tr.stats.endtime + 7 * 86400)

        if filtered_inv:
            last_epoch = filtered_inv[0][0][0]
            last_epoch.start_date = tr.stats.starttime - 1.0
            last_epoch.end_date = tr.stats.endtime + 1.0

            tr.remove_response(inventory=filtered_inv, **kwargs)
        else:
            raise ValueError


def remove_sensitivity_stable(tr, inv, **kwargs):
    try:
        tr.remove_sensitivity(inv, **kwargs)
    except ValueError:
        filtered_inv = inv.select(
            location=tr.stats.location, channel=tr.stats.channel,
            starttime=tr.stats.starttime - 7 * 86400,
            endtime=tr.stats.endtime + 7 * 86400)

        if filtered_inv:
            last_epoch = filtered_inv[0][0][0]
            last_epoch.start_date = tr.stats.starttime - 1.0
            last_epoch.end_date = tr.stats.endtime + 1.0

            tr.remove_sensitivity(inventory=filtered_inv, **kwargs)
        else:
            raise ValueError


def correct_subsample_shift(st):
    if len(st) > 1:
        shift = np.zeros(3)
        for i in range(1, 3):
            shift[i] = (st[i].stats.starttime - st[0].stats.starttime) % \
                       st[0].stats.delta

        if shift.sum() > 0.01:
            starttime = utct(0)
            endtime = utct()
            for tr in st:
                starttime = utct(max(float(starttime),
                                     float(tr.stats.starttime)))
                endtime = utct(min(float(endtime),
                                   float(tr.stats.endtime)))
            print(st)
            st.resample(tr.stats.sampling_rate * 10, no_filter=True)
            print(st)
            st.trim(starttime=starttime, endtime=endtime)
            print(st)
            st.decimate(5, no_filter=True)
            st.decimate(2, no_filter=True)
            print(st)


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


def calc_PSD(tr, winlen_sec, detick_nfsamp=0):
    Fs = tr.stats.sampling_rate

    if detick_nfsamp > 0:
        tr = detick(tr, detick_nfsamp)

    winlen = min(winlen_sec * Fs,
                 (tr.stats.endtime - tr.stats.starttime) * Fs / 2.)
    NFFT = next_pow_2(winlen)
    pad_to = np.max((NFFT * 2, 1024))
    p, f = mlab.psd(tr.data,
                    Fs=Fs, NFFT=NFFT, detrend='linear',
                    pad_to=pad_to, noverlap=NFFT // 2)
    return f, p


def detick(tr, detick_nfsamp, fill_val=None):
    # simplistic deticking by muting detick_nfsamp freqeuency samples around
    # 1Hz
    tr_out = tr.copy()
    Fs = tr.stats.sampling_rate
    NFFT = next_pow_2(tr.stats.npts)
    tr.detrend()
    df = np.fft.rfft(tr.data, n=NFFT)
    idx_1Hz = np.argmin(np.abs(np.fft.rfftfreq(NFFT) * Fs - 1.))
    if fill_val is None:
        fill_val = (df[idx_1Hz - detick_nfsamp - 1] + \
                    df[idx_1Hz + detick_nfsamp + 1]) / 2.
    df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] /= \
        df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] / fill_val
    tr_out.data = np.fft.irfft(df)[:tr.stats.npts]
    return tr_out


def plot_spectrum(ax, ax_all, df_mute, iax, ichan_in, spectrum,
                  fmin=0.1, fmax=100.,
                  **kwargs):
    f = spectrum['f']
    for i, chan in enumerate(['Z', 'N', 'E']):
        ichan = ichan_in + i
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

            bol_1Hz_mask = np.invert(bol_1Hz_mask)
            p = np.ma.masked_where(condition=bol_1Hz_mask, a=p,
                                   copy=False)
            f = np.ma.masked_where(condition=bol_1Hz_mask, a=f,
                                   copy=False)

            if ichan % 3 == 0:
                ax_all[ichan % 3].plot(f,
                                       10 * np.log10(p),
                                       lw=0.5, c='lightgrey', zorder=1)
                ax[iax, ichan].plot(f,
                                    10 * np.log10(p),
                                    **kwargs)
            elif ichan % 3 == 1:
                tmp2 = p
            elif ichan % 3 == 2:
                ax_all[ichan % 3 - 1].plot(f,
                                           10 * np.log10(tmp2 + p),
                                           lw=0.5, c='lightgrey', zorder=1)
                ax[iax, ichan - 1].plot(f,
                                        10 * np.log10(p + tmp2),
                                        **kwargs)

            # ax[iax, ichan].axes.get_xaxis().set_visible(False)
            # ax[iax, ichan].axes.get_yaxis().set_visible(False)
            ichan += 1


def envelope_smooth(envelope_window_in_sec, tr, mode='valid'):
    tr_env = tr.copy()
    tr_env.data = envelope(tr_env.data)

    w = np.ones(int(envelope_window_in_sec / tr.stats.delta))
    w /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, mode=mode)

    return tr_env


# Autocorrelation stuff

# def norm_hilbert(x, Fs):
#     x_white = whiten(x)
#     x_filt = filt(x_white, Fs=Fs, freqs=(1.5, 4.))
#     Z = hilbert(x_filt)
#     return Z / np.abs(Z)


def whiten(x):
    fx = fft(x, n=next_pow_2(len(x)))
    fx /= np.abs(fx)
    return ifft(fx, n=next_pow_2(len(x))).real[0:len(x)]


def inst_phase(x):
    Z = hilbert(x)
    return np.angle(Z)


def filt(x, Fs, freqs):
    from scipy.signal import filtfilt, butter

    b, a = butter(N=8, Wn=freqs[0] / (Fs / 2), btype='high')
    y = filtfilt(b, a, x)
    b, a = butter(N=8, Wn=freqs[1] / (Fs / 2), btype='low')
    y = filtfilt(b, a, y)
    return y


def phase_ac(x, Fs, maxlag_sec=8., nu=2.5):
    # Phase cross-correlation as defined in
    # Schimmel, M.(1999), Phase cross-correlations: design, comparisons and
    # applications, Bull.Seismol.Soc.Am., 89, 1366 - -1378.
    # This function implements eq. 4
    # Parameter nu was introduced later, e.g. eq. 2 in:
    # Schimmel, M., E. Stutzmann, and J. Gallart (2011), Using instantaneous
    # phase coherence for signal extraction from ambient noise data at a
    # local to a global scale,
    # Geophys. J. Int., 184, 494–506, doi:10.1111/j.1365-246X.2010.04861.x.
    maxlag = int(maxlag_sec * Fs)
    ac = np.zeros(maxlag)
    i = 0
    for ilag in range(1, maxlag):  # -maxlag//2, maxlag//2):
        # plusterm = np.abs(norm_hilbert(x[0:-ilag], Fs)
        #                   + norm_hilbert(x[ilag:], Fs))
        # minusterm = np.abs(norm_hilbert(x[0:-ilag], Fs)
        #                    - norm_hilbert(x[ilag:], Fs))

        A = np.exp(1.j * inst_phase(x[0:-ilag]))
        B = np.exp(1.j * inst_phase(x[ilag:]))
        plusterm = np.abs(A + B)
        minusterm = np.abs(A - B)

        ac[i] = 1. / (2 * len(x)) * np.sum(plusterm ** nu - minusterm ** nu) * \
                np.sqrt(ilag / Fs)
        i += 1
    return ac


def autocorrelation(st, starttime, endtime, fmin=1.2, fmax=3.5, max_lag_sec=40):
    # st.decimate(2)

    Fs = int(st[0].stats.sampling_rate)
    max_lag = max_lag_sec * Fs

    fig, ax = plt.subplots(nrows=4, ncols=1, sharey='all', sharex='all',
                           figsize=(15, 8))

    freqs = [[1.1, 3.5],
             [1.1, 5.0],
             [1.1, 8.0],
             [3.0, 6.0]]
    for i, freq in enumerate(freqs):
        print(freq)
        st_work = st.copy()
        st_work.filter('highpass', freq=1. / 10., zerophase=True)
        st_work.filter('lowpass', freq=8., zerophase=True)
        st_work.trim(starttime=starttime,
                     endtime=endtime)
        st_work.taper(max_percentage=0.05)
        acsum = np.zeros((max_lag, 4))
        for tr in st_work:  # .select(channel='BHZ'):
            data = whiten(tr.data)
            data = filt(data, Fs=tr.stats.sampling_rate,
                        freqs=(freq[0], freq[1]))
            ac = phase_ac(data,
                          Fs=tr.stats.sampling_rate,
                          maxlag_sec=max_lag_sec)
            t_ac = np.arange(0, len(ac)) / Fs
            ax[i].plot(t_ac,
                       filt(ac, Fs=tr.stats.sampling_rate,
                            freqs=(fmin, fmax)),
                       lw=2, label=tr.stats.channel)
            acsum[:, i] += filt(ac, Fs=tr.stats.sampling_rate,
                                freqs=(fmin, fmax))

            # ac_CC = np.correlate(tr.data, tr.data, mode='same') \
            #         / (np.sum(tr.data * tr.data))
            # ax[1].plot(np.arange(-len(ac_CC) / 2, len(ac_CC) / 2) / Fs,
            #            ac_CC, lw=2, label=tr.stats.channel)
            # acsum_CC += ac_CC
        ax[i].plot(t_ac, acsum[:, i], lw=2, c='k',
                   label='Sum')
    # ax[0].plot(np.arange(0, len(acsum)) / Fs, abs(hilbert(acsum)),
    #            label='Env. of Sum',
    #            lw=2, c='r')
    ax[0].legend()
    # ax[1].plot(np.arange(-len(acsum_CC) / 2, len(acsum_CC) / 2) / Fs,
    #            acsum_CC, lw=2, c='k')
    # ax[1].plot(np.arange(-len(acsum_CC) / 2, len(acsum_CC) / 2) / Fs,
    #            abs(hilbert(acsum_CC)), lw=2, c='r')
    ax[1].set_xlabel('seconds')
    ax[0].set_ylim(-1.2, 1.2)
    ax[1].set_ylim(-1.2, 1.2)
    ax[0].set_xlim((0, 20))
    for a in ax:
        a.set_xticks(np.arange(0, 30), minor=True)
        a.grid('on', which='major')
        a.grid('on', which='minor', ls='dashed', color='grey')

    ax[0].set_title('Phase autocorrelation')
    # ax[1].set_title('CC autocorrelation')
    return fig, ax


def linregression(x: np.array, y: np.array, q: float = 0.95) -> tuple:
    # Do a linear regression for value pairs X, Y and return error estimate
    # for slope and intercept
    from scipy import stats
    n = len(x)
    slope, intercept, r_value, p_value, slope_err = stats.linregress(x, y)

    intercept_err = slope_err * np.sqrt(1. / n * np.sum(x * x))

    tstar = stats.t.ppf(q=q, df=n - 2)

    return (intercept, intercept_err * tstar, slope, slope_err * tstar)


def calc_specgram(tr, fmin=1. / 50, fmax=1. / 2, w0=16):
    from matplotlib.mlab import specgram
    dt = tr.stats.delta

    s, f, t = specgram(x=tr.data, NFFT=512, Fs=tr.stats.sampling_rate,
                       noverlap=256, pad_to=1024)

    # t = create_timevector(tr)
    f_bol = np.asarray(((fmin < f),
                        (f < fmax))).all(axis=0)

    return s[f_bol, :], f[f_bol], t


def calc_cwf(tr, fmin=1. / 50, fmax=1. / 2, w0=16):
    from obspy.signal.tf_misfit import cwt
    dt = tr.stats.delta

    scalogram = abs(cwt(tr.data, dt, w0=w0, nf=200,
                        fmin=fmin, fmax=fmax))

    # t = create_timevector(tr)
    t = np.linspace(0, dt * tr.stats.npts, tr.stats.npts)
    f = np.logspace(np.log10(fmin),
                    np.log10(fmax),
                    scalogram.shape[0])
    return scalogram ** 2, f, t


def create_timevector(tr, utct=False):
    timevec = [utct(t +
                    float(tr.stats.starttime)).datetime
               for t in tr.times()]
    return timevec
