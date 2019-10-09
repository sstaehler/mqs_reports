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
from plotly.subplots import make_subplots


def make_report(event, fnam_out):
    fig = make_subplots(rows=3, cols=2,
                        shared_xaxes=True,
                        specs=[[{"rowspan": 3}, {}],
                               [None, {}],
                               [None, {}]],
                        print_grid=True,
                        # subplot_titles=(
                        #    "The big thing",
                        #    "Mb picks",
                        #    "M2.4 picks",
                        #    "something small"
                        # )

                        )
    pick_plot(event, fig, types=['mb_P', 'mb_S'], row=1, col=2,
              )
    pick_plot(event, fig, types=['m2.4'], row=2, col=2,
              )
    pick_plot(event, fig, types=['m2.4'], row=3, col=2,
              )
    plot_spec(event, fig, row=1, col=1)
    import plotly.io as pio
    pio.show(fig)
    pio.write_html(fig, file='tmp/plotly.html')
    fig.write_image("tmp/plotly.pdf")


def plot_spec(event, fig, row, col, **kwargs):
    colors = ['black', 'aliceblue', 'coral', 'orange']
    for kind, color in zip(['noise', 'all', 'P', 'S'], colors):
        fig.add_trace(
            go.Scatter(x=event.spectra[kind]['f'],
                       y=10 * np.log10(event.spectra[kind]['p_Z']),
                       name=kind,
                       line=go.scatter.Line(color=color),
                       mode="lines", **kwargs),
            row=row, col=col)


def pick_plot(event, fig, types, row, col, **kwargs):
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
    fmin = freqs[types[0]][0]
    fmax = freqs[types[0]][1]

    tr.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)
    timevec = [utct(t +
                    float(tr.stats.starttime)).strftime('%Y-%m-%d %H:%M:%S.%f')
               for t in tr.times()]
    fig.add_trace(
        go.Scatter(x=timevec,  # tr.times() + float(tr.stats.starttime),
                   y=tr.data,
                   name='time series %s' % types[0],
                   mode="lines", **kwargs),
        row=row, col=col)
    for pick_type in types:  # ('Peak_MbP', 'Peak_MbS'):
        pick = pick_name[pick_type]
        tmin = utct(event.picks[pick]) - 10.
        tmax = utct(event.picks[pick]) + 10.
        tr_pick = tr.slice(starttime=tmin, endtime=tmax)
        timevec = [utct(t +
                        float(tr_pick.stats.starttime)).strftime(
            '%Y-%m-%d %H:%M:%S.%f')
            for t in tr_pick.times()]
        fig.add_trace(go.Scatter(x=timevec,
                                 y=tr_pick.data,
                                 name='pick window %s' % pick_type,
                                 mode="lines",
                                 line=go.scatter.Line(color="red"),
                                 **kwargs),
                      row=row, col=col)
    # fig.update_xaxes(title_text='time', row=row, col=col)
    fig.update_yaxes(title_text='displacement', row=row, col=col)


if __name__ == '__main__':
    from mqs_reports.catalog import Catalog

    events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191007.xml',
                     type_select='all', quality=('A', 'B', 'C'))
    inv = obspy.read_inventory('./mqs_reports/data/inventory.xml')
    events.read_waveforms(inv=inv, kind='DISP', sc3dir='/mnt/mnt_sc3data')
    events.calc_spectra(winlen_sec=10.)
    for name, event in events.events:
        print(name)
        try:
            events.events['S0260a'].make_report(fnam_out='./tmp/plotly.html')
        except(TypeError):
            print('Problem')
