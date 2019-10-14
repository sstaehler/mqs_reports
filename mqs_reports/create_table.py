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
    row = '    <tr>\n'
    if extras is None:
        for li, fmt in zip(list, fmts):
            if li is None:
                row += '<td>' + str(li) + '</td>\n'
            else:
                row += '<td>' + fmt % (li) + '</td>\n'
    else:
        for li, fmt, extra in zip(list, fmts, extras):
            if li is None:
                row += '<td>' + str(li) + '</td>\n'
            else:
                if extra is None:
                    row += '<td>' + fmt % (li) + \
                           '</td>\n'
                else:
                    try:
                        row += 8 * ' ' + '<td sorttable_customkey="%d">' % extra \
                               + fmt % (li) + '</td>\n'
                    except(ValueError):
                        row += 8 * ' ' + '<td sorttable_customkey=0>' + \
                               fmt % (li) + '</td>\n'

    row += '</tr>\n'
    return row


def write_html(catalog, fnam_out):
    output = create_header((' ',
                            'name',
                            'type',
                            'LQ',
                            'Time (UTC)',
                            'Time (LMST)',
                            'duration',
                            'distance',
                            'P-amplitude',
                            'S-amplitude',
                            '2.4 Hz pick',
                            '2.4 Hz fit',
                            'A0',
                            'MbP',
                            'MbS',
                            'M2.4',
                            'MFB'))
    formats = ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%3.1f',
               '%8.3E', '%8.3E', '%8.3E', '%8.3E', '%8.3E',
               '%3.1f', '%3.1f', '%3.1f', '%3.1f')
    ievent = len(catalog.events)
    for event_name, event in catalog.events.items():
        duration = utct(utct(event.picks['end']) -
                        utct(event.picks['start'])).strftime('%M:%S')
        utc_time = utct(event.picks['start']).strftime('%Y-%j %H:%M:%S')
        lmst_time = solify(utct(event.picks['start'])).strftime('%H:%M:%S')
        sortkey = (ievent,
                   None,
                   None,
                   None,
                   float(utct(event.picks['start'])),
                   None,
                   None,
                   None,
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
                   None
                   )

        link_report = \
            '<a href="%s" target="_blank">%s</a>'

        output += create_row(
            (ievent,
             link_report % (event.fnam_report, event_name),
             event.mars_event_type_short,
             event.quality,
             utc_time,
             lmst_time,
             duration,
             event.distance,
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
             event.magnitude(type='mb_P', distance=30.),
             event.magnitude(type='mb_S', distance=30.),
             event.magnitude(type='m2.4', distance=20.),
             event.magnitude(type='MFB', distance=20.)
             ),
            extras=sortkey,
            fmts=formats)
        ievent -= 1
    output += '</tbody>'
    footer = '        </table>\n    </body>\n</html>\n'
    output += footer
    with open(fnam_out, 'w') as f:
        f.write(output)



def create_header(column_names):
    header = '<!DOCTYPE html>\n' + \
             '<html>\n' + \
             '<head>\n' + \
             '  <script src="sorttable.js"></script>' + \
             '  <link rel="stylesheet" type="text/css" href="./table.css">' + \
             '</head>\n' + \
             '  <body>\n'
    output = header
    output += '<h1>MQS events until %s</h1>' % obspy.UTCDateTime()
    table_head = '  <table class="sortable" id="events">\n' + \
                 '    <thead>\n' + \
                 create_row(
                     column_names) + \
                 '    </thead>'
    output += table_head
    output += '  <tbody>'
    return output


def define_arguments():
    helptext = 'Create HTML overview table and individual event plots'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

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
    import warnings


    args = define_arguments()
    catalog = Catalog(fnam_quakeml=args.input_quakeml,
                      type_select=args.types, quality=args.quality)
    inv = obspy.read_inventory(args.inventory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalog.read_waveforms(inv=inv, kind='DISP', sc3dir=args.sc3_dir)
    catalog.calc_spectra(winlen_sec=20.)
    catalog.make_report(dir_out='reports')
    catalog.write_table(fnam_out='./overview.html')
