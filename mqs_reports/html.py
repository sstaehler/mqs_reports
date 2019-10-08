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

import mqs_reports.magnitudes as mag


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
                    row += '<td sorttable_customkey="%d">' % extra + \
                           fmt % (li) + '</td>\n'
    row += '</tr>\n'
    return row


def write_html(catalog):
    output = create_header(('name',
                            'Time (UTC)',
                            'Time (LMST)',
                            'duration',
                            'P-amplitude',
                            'S-amplitude',
                            '2.4 Hz amplitude',
                            'M$_{2.4}$'))
    formats = ('%s', '%s', '%s', '%s', '%8.3E', '%8.3E', '%8.3E', '%3.1f')
    for event_name, event in catalog.events.items():
        duration = utct(utct(event.picks['end']) -
                        utct(event.picks['start'])).strftime('%M:%S')
        utc_time = utct(event.picks['start']).strftime('%Y-%j')
        lmst_time = solify(utct(event.picks['start'])).strftime('%jM%H:%M:%S')
        sortkey = (None,
                   float(utct(event.picks['start'])),
                   float(utct(event.picks['start'])),
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
                   None
                   )

        output += create_row(
            (event_name,
             utc_time,
             lmst_time,
             duration,
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
             mag.M2_4(event.pick_amplitude('Peak_M2.4',
                                           comp='vertical',
                                           fmin=2.2, fmax=2.6),
                      distance=5.),
             ),
            extras=sortkey,
            fmts=formats)
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

events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191004.xml',
                 type_select='lower', quality=('A', 'B', 'C'))
inv = obspy.read_inventory('./mqs_reports/data/inventory.xml')
events.read_waveforms(inv=inv, kind='DISP', sc3dir='/mnt/mnt_sc3data')
events.write_table()
