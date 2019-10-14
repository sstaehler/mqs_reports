#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import numpy as np
import obspy
import plotly.graph_objects as go
from obspy import UTCDateTime as utct
from obspy.signal.filter import envelope
from plotly.subplots import make_subplots

from mqs_reports.magnitudes import lorenz


def make_report(event, fnam_out, annotations):
    fig = make_subplots(rows=3, cols=2,
                        shared_xaxes=True,
                        specs=[[{"rowspan": 3}, {}],
                               [None, {}],
                               [None, {}]],
                        subplot_titles=(
                            "Event spectrum",
                            "Mb picks",
                            "M2.4 picks",
                            "something small")
                        )
    pick_plot(event, fig, types=['mb_P', 'mb_S'], row=1, col=2,
              annotations=annotations
              )
    pick_plot(event, fig, types=['m2.4'], row=2, col=2,
              annotations=annotations
              )
    pick_plot(event, fig, types=['m2.4'], row=3, col=2,
              annotations=annotations
              )
    plot_spec(event, fig, row=1, col=1)

    fig.update_layout({"title": {"text": "Event %s overview" % event.name,
                                 "font": {"size": 30}}})

    import plotly.io as pio
    # pio.show(fig)
    pio.write_html(fig, file=fnam_out,
                   include_plotlyjs=True)
    event.fnam_report = fnam_out


def plot_spec(event, fig, row, col, ymin=-250, ymax=-170,
              df_mute=1.07, **kwargs):
    colors = ['black', 'navy', 'coral', 'orange']

    fmins = [0.1, 7.5]
    fmaxs = [7.5, 50]
    specs = [event.spectra, event.spectra_SP]
    for spec, fmin, fmax in zip(specs, fmins, fmaxs):
        if len(spec) > 0:
            for kind, color in zip(['noise', 'all', 'P', 'S'], colors):
                if spec[kind] is not None and 'f' in spec[kind]:
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

    amps = event.amplitudes
    if 'A0' in amps:
        A0 = amps['A0']
        tstar = amps['tstar']
        if A0 is not None and tstar is not None:
            fig.add_trace(
                go.Scatter(x=f,
                           y=A0 + f * tstar,
                           name='fit, %ddB, t*=%4.2f' % (A0, -tstar * 0.1),
                           line=go.scatter.Line(color='blue', width=2),
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

    if amps['A_24'] is not None:
        fig.add_trace(
            go.Scatter(x=f,
                       y=lorenz(f, A=amps['A_24'],
                                x0=amps['f_24'],
                                xw=amps['width_24']),
                       name='fit, %ddB, f0*=%4.2fHz' %
                            (amps['A_24'], amps['f_24']),
                       line=go.scatter.Line(color='darkblue', width=2),
                       mode="lines", **kwargs),
            row=row, col=col)
        # Add text marker
        fig.add_trace(
            go.Scatter(x=[2.3, 2.5],
                       y=[amps['A_24'], amps['A_24']],
                       showlegend=False,
                       text=['', 'A0=%d dB' % amps['A_24']],
                       textfont={'size': 20},
                       line=go.scatter.Line(color='blue', width=2),
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
                 'm2.4': 'Peak_M2.4'
                 }
    freqs = {'mb_P': (1. / 6., 1. / 2.),
             'mb_S': (1. / 6., 1. / 2.),
             'm2.4': (2., 3.)
             }
    component = {'mb_P': 'vertical',
                 'mb_S': 'horizontal',
                 'm2.4': 'vertical'}

    tr = event.waveforms_VBB.select(channel='??Z')[0].copy()
    tr.decimate(2)
    fmin = freqs[types[0]][0]
    fmax = freqs[types[0]][1]

    tr.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)
    tr.trim(starttime=utct(event.picks['start']) - 180.,
            endtime=utct(event.picks['end']) + 180.)
    env = envelope(tr.data)
    timevec = _create_timevector(tr)
    fig.add_trace(
        go.Scatter(x=timevec,
                   y=tr.data,
                   name='time series %s' % types[0],
                   line=go.scatter.Line(color="darkgrey"),
                   mode="lines", **kwargs),
        row=row, col=col)
    fig.add_trace(
        go.Scatter(x=timevec,
                   y=env,
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
                                         line=go.scatter.Line(color="cyan"),
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

    fig.update_yaxes(title_text='displacement', row=row, col=col)


def _create_timevector(tr):
    strf = '%Y-%m-%d %H:%M:%S.%f'
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
    for name, event in events.events.items():
        event.make_report(fnam_out='./reports/plotly_%s.html' % name)
