#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from argparse import ArgumentParser

import obspy
from mars_tools.insight_time import solify
from obspy import UTCDateTime as utct
from tqdm import tqdm

from mqs_reports.snr import calc_SNR, calc_stalta


def create_row_header(list):
    row = '    <tr>\n'
    for li in zip(list):
        row += '<th>' + str(li) + '</th>\n'
    row += '</tr>\n'
    return row


def create_row(list, fmts=None, extras=None):
    if fmts is None:
        fmts = []
        for i in range(len(list)):
            fmts.append('%s')
    row = 4 * ' ' +'<tr>\n'
    ind_string = 6 * ' '
    if extras is None:
        for li, fmt in zip(list, fmts):
            if li is None:
                row += ind_string + '<td>-</td>\n'
            else:
                row += ind_string + '<td>' + fmt % (li) + '</td>\n'
    else:
        for li, fmt, extra in zip(list, fmts, extras):
            if li is None:
                row += ind_string + '<td>-</td>\n'
            else:
                if extra is None:
                    row += ind_string + '<td>' + fmt % (li) + \
                           '</td>\n'
                else:
                    try:
                        row += ind_string \
                               + '<td sorttable_customkey="%d">' % extra \
                               + fmt % (li) + '</td>\n'
                    except(ValueError):
                        row += ind_string + '<td sorttable_customkey=0>' + \
                               fmt % (li) + '</td>\n'

    row += 4 * ' ' + '</tr>\n'
    return row


def write_html(catalog, fnam_out):
    output = create_html_header()
    output += catalog.get_event_count_table()
    output += create_table_head(
        column_names=(' ',
                      'name',
                      'type',
                      'LQ',
                      'Time<br>(UTC)',
                      'Time<br>(LMST)',
                      'duration<br>[minutes]',
                      'distance<br>[degree]',
                      'SNR',
                      'P-amp<br>[m]',
                      'S-amp<br>[m]',
                      '2.4 Hz<br>pick [m]',
                      '2.4 Hz<br>fit [m]',
                      'A0<br>[m]',
                      'MbP',
                      'MbS',
                      'M2.4',
                      'MFB',
                      'f_c<br>[Hz]',
                      'tstar<br>[s]',
                      'VBB<br>rate',
                      '100sps<br> SP1',
                      '100sps<br> SPH'))
    formats = ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
               '%s', '%8.2E', '%8.2E', '%8.2E', '%8.2E', '%8.2E',
               '%3.1f', '%3.1f', '%3.1f', '%3.1f', '%3.1f', '%5.3f',
               '%s', '%s', '%s')
    dist_string = {'GUI': '%.3g',
                   'PgSg': '%.3g*',
                   'aligned': '%.3g&dagger;',
                   'unknown': '%s'}
    event_type_idx = {'LF': 1,
                      'BB': 2,
                      'HF': 3,
                      '24': 4,
                      'VF': 5,
                      'UF': 6}
    ievent = len(catalog)
    print('Filling HTML table with event entries')
    for event in tqdm(catalog):
        utc_time = event.starttime.strftime('%Y-%m-%d<br>%H:%M:%S')
        lmst_time = solify(event.starttime).strftime('%H:%M:%S')
        duration = event.duration.strftime('%M:%S')
        sortkey = (ievent,
                   None,
                   event_type_idx[event.mars_event_type_short],
                   None,
                   float(utct(event.picks['start'])),
                   None,
                   None,
                   None,
                   calc_SNR(event, fmin=2.1, fmax=2.7)
                   if event.mars_event_type_short in ('HF', '24')
                       else calc_SNR(event, fmin=0.2, fmax=0.5),
                   event.pick_amplitude('Peak_MbP',
                                        comp='vertical',
                                        fmin=1. / 6.,
                                        fmax=1. / 2,
                                        unit='fm'),
                   event.pick_amplitude('Peak_MbS',
                                        comp='horizontal',
                                        fmin=1. / 6.,
                                        fmax=1. / 2,
                                        unit='fm'),
                   event.pick_amplitude('Peak_M2.4',
                                        comp='vertical',
                                        fmin=2.2, fmax=2.6,
                                        unit='fm'),
                   event.amplitudes['A_24'],
                   event.amplitudes['A0'],
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None
                   )
        event.fnam_report['name'] = event.name
        link_report = \
            ('<a href="{Z:s}" target="_blank">{name:s}</a><br>' +
             '<a href="{Z:s}" target="_blank">Z</a> ' +
             '<a href="{N:s}" target="_blank">N</a> ' +
             '<a href="{E:s}" target="_blank">E</a>').format(
                **event.fnam_report)
        # snr_string = '%.1f (2.4 Hz)' % calc_SNR(event, fmin=2.1, fmax=2.7) \
        if event.mars_event_type_short in ('HF', 'VF', '24'):
            snr_string = '%.1f (2.4Hz)' % calc_stalta(event,
                                                      fmin=2.2, fmax=2.8)
        elif event.mars_event_type_short == ('UF'):
            snr_string = '%.1f (8-12Hz)' % calc_SNR(event, fmin=8.0, fmax=12.,
                                                    SP=True, hor=True)
        else:
            snr_string = '%.1f (2-5s)' % calc_SNR(event, fmin=0.2, fmax=0.5)

        output += create_row(
            (ievent,
             link_report,
             event.mars_event_type_short,
             event.quality,
             utc_time,
             lmst_time,
             duration,
             dist_string[event.distance_type] % event.distance,
             snr_string,
             event.pick_amplitude('Peak_MbP',
                                  comp='vertical',
                                  fmin=1. / 6.,
                                  fmax=1. / 2),
             event.pick_amplitude('Peak_MbS',
                                  comp='horizontal',
                                  fmin=1. / 6.,
                                  fmax=1. / 2),
             event.pick_amplitude('Peak_M2.4',
                                  comp='vertical',
                                  fmin=2.2, fmax=2.6),
             10 ** (event.amplitudes['A_24'] / 20.)
             if event.amplitudes['A_24'] is not None else None,
             10 ** (event.amplitudes['A0'] / 20.)
             if event.amplitudes['A0'] is not None else None,
             event.magnitude(mag_type='mb_P'),
             event.magnitude(mag_type='mb_S'),
             event.magnitude(mag_type='m2.4'),
             event.magnitude(mag_type='MFB'),
             event.amplitudes['f_c'],
             event.amplitudes['tstar'],
             event.available_sampling_rates()['VBB_Z'],
             _fmt_bool(event.available_sampling_rates()['SP_Z'] == 100.),
             _fmt_bool(event.available_sampling_rates()['SP_N'] == 100.),
             ),
            extras=sortkey,
            fmts=formats)
        ievent -= 1
    footer = create_footer()
    output += footer
    with open(fnam_out, 'w') as f:
        f.write(output)


def create_footer():
    footer = 4 * ' ' + '</tbody>\n'
    footer += 2 * ' ' + '</table>\n</article>\n</body>\n</html>\n'
    return footer


def _fmt_bool(bool):
    if bool:
        return '&#9745;'
    else:
        return ' '


def create_html_header():
    header = '<!DOCTYPE html>\n' + \
             '<html lang="en-US">\n' + \
             '<head>\n' + \
             '  <script src="sorttable.js"></script>\n' + \
             '  <title>MQS events until %s</title>\n' % utct().date + \
             '  <meta charset="UTF-8">\n' + \
             '  <meta name="description" content="InSight marsquakes">\n' + \
             '  <meta name="author" content="Marsquake Service" >\n' + \
             '  <link rel="stylesheet" type="text/css" href="./table.css">\n' + \
             '</head>\n' + \
             '<body>\n'
    output = header
    return output


def create_table_head(column_names):
    output = ''
    output += '<article>\n'
    output += '  <header>\n'
    output += '    <h1>Event table</h1>\n'
    output += '  </header>\n'
    table_head = '  <table class="sortable" id="events">\n' + \
                 '  <thead>\n' + \
                 create_row(column_names) + \
                 '  </thead>\n'
    output += table_head
    output += '  <tbody>\n'
    return output


def define_arguments():
    helptext = 'Create HTML overview table and individual event plots'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

    helptext = 'Input annotation file'
    parser.add_argument('input_csv', help=helptext)

    helptext = 'Input manual distance file'
    parser.add_argument('input_dist', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('inventory', help=helptext)

    helptext = 'Path to SC3DIR'
    parser.add_argument('sc3_dir', help=helptext)

    helptext = 'Location qualities (one or more)'
    parser.add_argument('-q', '--quality', help=helptext,
                        nargs='+', default=('A', 'B'))

    helptext = 'Event types'
    parser.add_argument('-t', '--types', help=helptext,
                        default='all')

    return parser.parse_args()


if __name__ == '__main__':
    from mqs_reports.catalog import Catalog
    from mqs_reports.annotations import Annotations
    import warnings


    args = define_arguments()
    catalog = Catalog(fnam_quakeml=args.input_quakeml,
                      type_select=args.types, quality=args.quality)
    ann = Annotations(fnam_csv=args.input_csv)

    # load manual (aligned) distances
    catalog.load_distances(fnam_csv=args.input_dist)
    inv = obspy.read_inventory(args.inventory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalog.read_waveforms(inv=inv, kind='DISP', sc3dir=args.sc3_dir)
    catalog.calc_spectra(winlen_sec=20.)

    catalog.make_report(dir_out='reports', annotations=ann)
    catalog.write_table(fnam_out='./overview.html')
