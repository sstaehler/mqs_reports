#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""
import numpy as np
import obspy
import plotly.graph_objects as go
import plotly.io as pio
from obspy import UTCDateTime as utct
from plotly.subplots import make_subplots

from mqs_reports.magnitudes import lorentz, lorentz_att
from mqs_reports.utils import envelope_smooth, detick, calc_cwf, \
    create_timevector


def make_report(event, chan, fnam_out, annotations):
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
              annotations=annotations, chan=chan)
    pick_plot(event, fig, types=['m2.4'], row=2, col=2,
              annotations=annotations, chan=chan)

    plot_specgram(event, fig, row=3, col=2, chan=chan)

    plot_spec(event, fig, row=1, col=1, chan=chan)

    fig.update_layout({"title": {"text": "Event %s overview BH%s" %
                                         (event.name, chan),
                                 "font": {"size": 30}}})

    pio.write_html(fig, file=fnam_out + '.html',
                   full_html=True,
                   include_plotlyjs='directory')

    fig.write_image(file=fnam_out + '.pdf', 
                    width=1200, height=int(900 * 0.75))
    event.fnam_report[chan] = fnam_out


def plot_specgram(event, fig, row, col, chan, fmin=0.05, fmax=10.0):
    if event.waveforms_VBB is not None:
        tr = detick(event.waveforms_VBB.select(channel='??' + chan)[0],
                    detick_nfsamp=10)
        tr.trim(starttime=utct(event.picks['start']) - 180.,
                endtime=utct(event.picks['end']) + 180.)

        tr.differentiate()
        tr.differentiate()
        z, f, t = calc_cwf(tr,
                           fmin=fmin, fmax=fmax)
        z = 10 * np.log10(z)
        z[z < -220] = -220.
        z[z > -150] = -150.
        df = 2
        dt = 4
        data_heatmap = go.Heatmap(z=z[::df, ::dt],
                                  x=t[::dt], y=f[::df],
                                  colorscale='plasma')
        data_heatmap.colorbar.len = 0.35
        data_heatmap.colorbar.yanchor = 'bottom'
        data_heatmap.colorbar.y = 0.0
        data_heatmap.colorbar.title.text = '(m/s²)²/Hz [dB]'
        fig.add_trace(data_heatmap,
                      row=row, col=col)

    for pick in ['start', 'end', 'P', 'S', 'Pg', 'Sg']:
        if event.picks[pick] != '':
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


def plot_spec(event,
              fig, row, col, chan,
              ymin=-250, ymax=-170,
              df_mute=0.99, f_VBB_SP_transition=7.5, **kwargs):
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
                    p = spec[kind]['p_' + chan]

                    fig.add_trace(
                        go.Scatter(x=f[bol_1Hz_mask],
                                   y=10 * np.log10(p[bol_1Hz_mask]),
                                   name=kind,
                                   legendgroup=kind,
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
        # Add only VBB text
        fig.add_trace(
            go.Scatter(x=[f_VBB_SP_transition,
                          f_VBB_SP_transition,
                          f_VBB_SP_transition],
                       y=[-250, -180, -100],
                       showlegend=False,
                       text=['', 'VBB' + chan, ''],
                       textfont={'size': 30},
                       textposition='bottom center',
                       mode="text", **kwargs),
            row=row, col=col)

    amps = event.amplitudes
    f = np.geomspace(0.1, 50.0, num=400)
    if 'A0' in amps:
        A0 = amps['A0']
        tstar = amps['tstar']
        f_c = amps['f_c'] if 'f_c' in amps and amps['f_c'] is not None else 3.
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
                               y=lorentz_att(f, A0=amps['A0'],
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
                       y=lorentz(f, A=amps['A_24'],
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
                     title_text='PSD, displacement / m²/Hz [dB]',
                     row=row, col=col)
    fig.update_xaxes(range=[-1, 1.5], type='log',
                     title_text='frequency / Hz',
                     row=row, col=col)


def pick_plot(event, fig, types, row, col, chan, annotations=None, **kwargs):
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

    if ((event.waveforms_SP is None or len(event.waveforms_SP) == 0) 
       and 
       (event.waveforms_VBB is None or len(event.waveforms_VBB) == 0)):
        print('SP:') 
        print(event.waveforms_SP)        

        print('VBB:') 
        print(event.waveforms_VBB)        
        print('No data for event %s' % event.name)
    else:
        if event.waveforms_VBB is None:
            tr = event.waveforms_SP.select(channel='??' + chan)[0].copy()
        else:
            tr = event.waveforms_VBB.select(channel='??' + chan)[0].copy()
        tr.decimate(2)
        fmin = freqs[types[0]][0]
        fmax = freqs[types[0]][1]

        tr.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)
        tr.trim(starttime=utct(event.picks['start']) - 180.,
                endtime=utct(event.picks['end']) + 180.)
        tr_env = envelope_smooth(envelope_window_in_sec=60.,
                                 tr=tr)
        tr_env.stats.starttime += 30.
        tr_env.data *= 2.
        timevec = create_timevector(tr)
        fig.add_trace(
            go.Scatter(x=timevec,
                       y=tr.data,
                       name='time series %s' % types[0],
                       showlegend=False,
                       line=go.scatter.Line(color="darkgrey"),
                       mode="lines", **kwargs),
            row=row, col=col)
        timevec = create_timevector(tr_env)
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
                    timevec = create_timevector(tr_pick)
                    fig.add_trace(go.Scatter(x=timevec,
                                             y=tr_pick.data,
                                             showlegend=False,
                                             mode="lines",
                                             line=go.scatter.Line(
                                                 color="lightgrey"),
                                             **kwargs),
                                  row=row, col=col)
        # cols = dict(mb_P='red',
        #         mb_S=
        for pick_type in types:
            pick = pick_name[pick_type]
            if event.picks[pick] != '':
                tmin = utct(event.picks[pick]) - 10.
                tmax = utct(event.picks[pick]) + 10.
                tr_pick = tr.slice(starttime=tmin, endtime=tmax)
                timevec = create_timevector(tr_pick)
                fig.add_trace(go.Scatter(x=timevec,
                                         y=tr_pick.data,
                                         name='pick window %s' % pick_type,
                                         mode="lines",
                                         line=go.scatter.Line(),
                                         **kwargs),
                              row=row, col=col)

        for pick in ['start', 'end', 'P', 'S', 'Pg', 'Sg']:
            if event.picks[pick] != '':
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

    fig.update_yaxes(title_text='displacement / m', row=row, col=col)


if __name__ == '__main__':
    from mqs_reports.catalog import Catalog

    events = Catalog(fnam_quakeml='./mqs_reports/data/catalog.xml',
                     type_select='all', quality=('A', 'B', 'C'))
    inv = obspy.read_inventory('./mqs_reports/data/inventory.xml')
    events = events.select(name='S0376a')
    events.read_waveforms(inv=inv, kind='DISP', sc3dir='/mnt/mnt_sc3data')
    events.calc_spectra(winlen_sec=20.)
    events.make_report()
