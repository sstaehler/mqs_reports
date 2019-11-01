#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""
import mqs_reports
import numpy as np
import obspy
import plotly.graph_objects as go
from mqs_reports.magnitudes import lorenz, lorenz_att
from mqs_reports.utils import envelope_smooth
from obspy import UTCDateTime as utct
from plotly.subplots import make_subplots


def make_report(event, fnam_out, annotations):
    fig = make_subplots(rows=3, cols=2,
                        shared_xaxes=True,
                        specs=[[{"rowspan": 3}, {}],
                               [None, {}],
                               [None, {}]],
                        horizontal_spacing=0.05,
                        vertical_spacing=0.05,
                        subplot_titles=(
                            "<b>Event spectrum, vertical</b>",
                            "<b>Mb picks 1.5-6 sec</b>",
                            "<b>M2.4 picks 2-3 Hz</b>",
                            "<b>Acceleration spectrogram</b>")
                        )
    pick_plot(event, fig, types=['mb_P', 'mb_S'], row=1, col=2,
              annotations=annotations
              )
    pick_plot(event, fig, types=['m2.4'], row=2, col=2,
              annotations=annotations
              )
    # pick_plot(event, fig, types=['full', 'mb_P', 'mb_S'],
    #           row=3, col=2,
    #           annotations=annotations
    #           )

    plot_specgram(event, fig, row=3, col=2)

    plot_spec(event, fig, row=1, col=1)

    fig.update_layout({"title": {"text": "Event %s overview" % event.name,
                                 "font": {"size": 30}}})

    import plotly.io as pio
    # pio.show(fig)
    pio.write_html(fig, file=fnam_out,
                   include_plotlyjs=True)
    event.fnam_report = fnam_out


def plot_specgram(event, fig, row, col, fmin=0.05, fmax=10.0):
    tr = event.waveforms_VBB.select(channel='??Z')[0].copy()
    tr.trim(starttime=utct(event.picks['start']) - 180.,
            endtime=utct(event.picks['end']) + 180.)

    tr.differentiate()
    tr.differentiate()
    z, f, t = _calc_cwf(tr, fmin=fmin, fmax=fmax)
    z = 10 * np.log10(z)
    z[z < -220] = -220.
    z[z > -160] = -160.
    fig.add_trace(go.Heatmap(z=z[::4, ::4],
                             x=t[::4], y=f[::4],
                             colorscale='plasma'),
                  row=row, col=col)

    for pick in ['start', 'end', 'P', 'S', 'Pg', 'Sg']:
        if event.picks[pick] is not '':
            time_pick = utct(event.picks[pick]).datetime
            if pick not in ['start', 'end']:
                text = pick
                color = 'black'
            else:
                text = ''
                color = 'darkgreen'
            fig.add_trace(go.Scatter(x=[time_pick, time_pick],
                                     y=[-fmin, fmax],
                                     text=['', text],
                                     showlegend=False,
                                     textfont={'size': 20},
                                     textposition='bottom right',
                                     name=pick,
                                     mode="lines+text",
                                     line=go.scatter.Line(color=color,
                                                          width=0.5),
                                     ),
                          row=row, col=col)
    fig.update_yaxes(range=[np.log10(fmin), np.log10(fmax)],
                     type='log',
                     title_text='frequency / Hz',
                     row=row, col=col)


def plot_spec(event: mqs_reports.event.Event,
              fig, row, col, ymin=-250,
              ymax=-170,
              df_mute=1.07, f_VBB_SP_transition=7.5, **kwargs):
    colors = ['black', 'navy', 'coral', 'orange']

    fmins = [0.08, f_VBB_SP_transition]
    fmaxs = [f_VBB_SP_transition, 50]
    specs = [event.spectra, event.spectra_SP]
    for spec, fmin, fmax in zip(specs, fmins, fmaxs):
        if len(spec) > 0:
            for kind, color in zip(['noise', 'all', 'P', 'S'], colors):
                if kind in spec and 'f' in spec[kind]:
                    f = spec[kind]['f']
                    bol_1Hz_mask = np.array(
                        (np.array((f >= fmin, f <= fmax)).all(axis=0),
                         np.array((f < 1. / df_mute,
                                   f > df_mute)).any(axis=0))
                        ).all(axis=0)
                    p = spec[kind]['p_Z']

                    fig.add_trace(
                        go.Scatter(x=f[bol_1Hz_mask],
                                   y=10 * np.log10(p[bol_1Hz_mask]),
                                   name=kind,
                                   line=go.scatter.Line(color=color),
                                   mode="lines", **kwargs),
                        row=row, col=col)

    if event.waveforms_SP is not None:
        # Add marker for SP/VBB transition
        fig.add_trace(
            go.Scatter(x=[f_VBB_SP_transition,
                          f_VBB_SP_transition,
                          f_VBB_SP_transition],
                       y=[-250, -180, -100],
                       showlegend=False,
                       text=['', 'VBB  SP1', ''],
                       textfont={'size': 30},
                       line=go.scatter.Line(color='LightSeaGreen',
                                            dash='dashdot',
                                            width=2),
                       textposition='bottom center',
                       mode="lines+text", **kwargs),
            row=row, col=col)
    else:
        # Add only VBBZ text
        fig.add_trace(
            go.Scatter(x=[f_VBB_SP_transition,
                          f_VBB_SP_transition,
                          f_VBB_SP_transition],
                       y=[-250, -180, -100],
                       showlegend=False,
                       text=['', 'VBBZ', ''],
                       textfont={'size': 30},
                       textposition='bottom center',
                       mode="text", **kwargs),
            row=row, col=col)

    amps = event.amplitudes
    f = np.geomspace(0.1, 50.0, num=400)
    if 'A0' in amps:
        A0 = amps['A0']
        tstar = amps['tstar']
        f_c = amps['f_c'] if amps['f_c'] is not None else 3.
        if A0 is not None and tstar is not None:
            stf_amp = 1 / (1 + (f / f_c) ** 2)
            fig.add_trace(
                go.Scatter(x=f,
                           y=A0 - f * tstar * 10. * np.pi / np.log(10)
                             + 10 * np.log10(stf_amp),
                           name='fit: source, att.',
                           line=go.scatter.Line(color='blue', width=2),
                           mode="lines", **kwargs),
                row=row, col=col)
            # Fit Lorenz-Adapted tstar
            if amps['ampfac'] is not None:
                fig.add_trace(
                    go.Scatter(x=f,
                               y=lorenz_att(f, A0=amps['A0'],
                                            f0=amps['f_24'],
                                            f_c=amps['f_c'],
                                            tstar=amps['tstar'],
                                            fw=amps['width_24'],
                                            ampfac=amps['ampfac']),
                               name='fit: src, att, amplific.<br>'
                                    '%ddB, f=%4.2fHz, f_c=%4.2fHz<br>'
                                    't*=%4.2f, df=%4.2f, dA=%4.1fdB' %
                                    (amps['A0'], amps['f_24'], amps['f_c'],
                                     amps['tstar'],
                                     amps['width_24'],
                                     10 * np.log10(amps['ampfac'])),
                               line=go.scatter.Line(color='red', width=5),
                               mode="lines", **kwargs),
                    row=row, col=col)
            # Add text marker
            fig.add_trace(
                go.Scatter(x=[0.05, 0.15],
                           y=[A0, A0],
                           showlegend=False,
                           text=['', 'A0=%d dB' % A0],
                           textfont={'size': 20},
                           line=go.scatter.Line(color='blue', width=2),
                           textposition='bottom right',
                           mode="lines+markers+text", **kwargs),
                row=row, col=col)

    if 'A_24' in amps and amps['A_24'] is not None:
        fig.add_trace(
            go.Scatter(x=f,
                       y=lorenz(f, A=amps['A_24'],
                                x0=amps['f_24'],
                                xw=amps['width_24']),
                       name='fit, peak only<br>'
                            '%ddB, f0*=%4.2fHz' %
                            (amps['A_24'], amps['f_24']),
                       line=go.scatter.Line(color='darkblue', width=2),
                       mode="lines", **kwargs),
            row=row, col=col)
        # Add text marker
        fig.add_trace(
            go.Scatter(x=[2.3, 2.5],
                       y=[amps['A_24'], amps['A_24']],
                       line=go.scatter.Line(color='blue', width=2),
                       showlegend=False,
                       text=['', 'A_24=%d dB' % amps['A_24']],
                       textfont={'size': 20},
                       textposition='bottom right',
                       mode="lines+markers+text", **kwargs),
            row=row, col=col)

    fig.update_yaxes(range=[ymin, ymax],
                     title_text='PSD, displacement / (m/s²)²/Hz [dB]',
                     row=row, col=col)
    fig.update_xaxes(range=[-1, 1.5], type='log',
                     title_text='frequency / Hz',
                     row=row, col=col)


def pick_plot(event, fig, types, row, col, annotations=None, **kwargs):
    pick_name = {'mb_P': 'Peak_MbP',
                 'mb_S': 'Peak_MbS',
                 'm2.4': 'Peak_M2.4',
                 'full': 'Peak_MbP'
                 }
    freqs = {'mb_P': (1. / 6., 1. / 2.),
             'mb_S': (1. / 6., 1. / 2.),
             'm2.4': (2., 3.),
             'full': (1. / 15., 3.5)
             }

    tr = event.waveforms_VBB.select(channel='??Z')[0].copy()
    tr.decimate(2)
    fmin = freqs[types[0]][0]
    fmax = freqs[types[0]][1]

    tr.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)
    tr.trim(starttime=utct(event.picks['start']) - 180.,
            endtime=utct(event.picks['end']) + 180.)
    tr_env = envelope_smooth(envelope_window=60.,
                             tr=tr)
    tr_env.stats.starttime += 30.
    tr_env.data *= 2.
    timevec = _create_timevector(tr)
    fig.add_trace(
        go.Scatter(x=timevec,
                   y=tr.data,
                   name='time series %s' % types[0],
                   line=go.scatter.Line(color="darkgrey"),
                   mode="lines", **kwargs),
        row=row, col=col)
    timevec = _create_timevector(tr_env)
    fig.add_trace(
        go.Scatter(x=timevec,
                   y=tr_env.data,
                   name='envelope %s' % types[0],
                   showlegend=False,
                   line=go.scatter.Line(color="darkgrey", dash='dot'),
                   mode="lines", **kwargs),
        row=row, col=col)

    if annotations is not None:
        annotations_event = annotations.select(
            starttime=utct(event.picks['start']) - 180.,
            endtime=utct(event.picks['end']) + 180.)
        if len(annotations_event) > 0:
            for times in annotations_event:
                tmin = utct(times[0])
                tmax = utct(times[1])
                tr_pick = tr.slice(starttime=tmin, endtime=tmax)
                timevec = _create_timevector(tr_pick)
                fig.add_trace(go.Scatter(x=timevec,
                                         y=tr_pick.data,
                                         showlegend=False,
                                         mode="lines",
                                         line=go.scatter.Line(
                                             color="lightgrey"),
                                         **kwargs),
                              row=row, col=col)

    for pick_type in types:
        pick = pick_name[pick_type]
        if event.picks[pick] is not "":
            tmin = utct(event.picks[pick]) - 10.
            tmax = utct(event.picks[pick]) + 10.
            tr_pick = tr.slice(starttime=tmin, endtime=tmax)
            timevec = _create_timevector(tr_pick)
            fig.add_trace(go.Scatter(x=timevec,
                                     y=tr_pick.data,
                                     name='pick window %s' % pick_type,
                                     mode="lines",
                                     line=go.scatter.Line(color="red"),
                                     **kwargs),
                          row=row, col=col)

    for pick in ['start', 'end', 'P', 'S', 'Pg', 'Sg']:
        if event.picks[pick] is not '':
            time_pick = utct(event.picks[pick]).datetime
            ymax = np.max(abs(tr.data))
            if pick not in ['start', 'end']:
                text = pick
                color = 'black'
            else:
                text = ''
                color = 'darkgreen'
            fig.add_trace(go.Scatter(x=[time_pick, time_pick],
                                     y=[-ymax, ymax],
                                     text=['', text],
                                     showlegend=False,
                                     textfont={'size': 20},
                                     textposition='bottom right',
                                     name=pick,
                                     mode="lines+text",
                                     line=go.scatter.Line(color=color),
                                     **kwargs),
                          row=row, col=col)

    fig.update_yaxes(title_text='displacement', row=row, col=col)


def _calc_cwf(tr, fmin=1. / 50, fmax=1. / 2, w0=16):
    from obspy.signal.tf_misfit import cwt
    npts = tr.stats.npts
    dt = tr.stats.delta

    scalogram = abs(cwt(tr.data, dt, w0=w0, nf=200,
                        fmin=fmin, fmax=fmax))

    t = _create_timevector(tr)  # np.linspace(0, dt * npts, npts)
    f = np.logspace(np.log10(fmin),
                    np.log10(fmax),
                    scalogram.shape[0])
    return scalogram ** 2, f, t

def _create_timevector(tr):
    timevec = [utct(t +
                    float(tr.stats.starttime)).datetime
               for t in tr.times()]
    return timevec


if __name__ == '__main__':
    from mqs_reports.catalog import Catalog

    events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191007.xml',
                     type_select='all', quality=('A', 'B', 'C'))
    inv = obspy.read_inventory('./mqs_reports/data/inventory.xml')
    events.read_waveforms(inv=inv, kind='DISP', sc3dir='/mnt/mnt_sc3data')
    events.calc_spectra(winlen_sec=20.)
