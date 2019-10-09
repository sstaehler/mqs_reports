#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

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


def write_html(catalog):
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
                            '2.4 Hz amplitude',
                            'MbP',
                            'MbS',
                            'M2.4'))
    formats = ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%3.1f',
               '%8.3E', '%8.3E', '%8.3E', '%3.1f', '%3.1f', '%3.1f')
    ievent = len(catalog.events)
    for event_name, event in catalog.events.items():
        duration = utct(utct(event.picks['end']) -
                        utct(event.picks['start'])).strftime('%M:%S')
        utc_time = utct(event.picks['start']).strftime('%Y-%j')
        lmst_time = solify(utct(event.picks['start'])).strftime('%jM%H:%M:%S')
        sortkey = (ievent,
                   None,
                   None,
                   None,
                   float(utct(event.picks['start'])),
                   float(utct(event.picks['start'])),
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
                   None,
                   None,
                   None
                   )

        output += create_row(
            (ievent,
             event_name,
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
             event.magnitude(type='mb_P', distance=30.),
             event.magnitude(type='mb_S', distance=30.),
             event.magnitude(type='m2.4', distance=20.)
             ),
            extras=sortkey,
            fmts=formats)
        ievent -= 1
    output += '</tbody>'
    footer = '        </table>\n    </body>\n</html>\n'
    output += footer
    with open('tmp/test.html', 'w') as f:
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


from mqs_reports.catalog import Catalog

events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191007.xml',
                 type_select='all', quality=('A', 'B', 'C', 'D'))
inv = obspy.read_inventory('./mqs_reports/data/inventory.xml')
events.read_waveforms(inv=inv, kind='DISP', sc3dir='/mnt/mnt_sc3data')
events.write_table()