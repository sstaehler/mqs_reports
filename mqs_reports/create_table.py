#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from argparse import ArgumentParser
from os.path import exists as pexists, join as pjoin

import obspy
from mqs_reports.snr import calc_SNR, calc_stalta
from mqs_reports.utils import solify
from obspy import UTCDateTime as utct
from tqdm import tqdm


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
            if li is None or (type(li) is tuple and not all(li)):
                row += ind_string + '<td>-</td>\n'
            else:
                if extra is None:
                    row += ind_string + '<td>' + fmt % (li) + \
                           '</td>\n'
                else:
                    try:
                        row += ind_string \
                               + '<td sorttable_customkey="%d">' % (extra * 100) \
                               + fmt % (li) + '</td>\n'
                    except(ValueError):
                        row += ind_string + '<td sorttable_customkey=-100000>' + \
                               fmt % (li) + '</td>\n'

    row += 4 * ' ' + '</tr>\n'
    return row

def add_information():
    string = '<H2>Distance types: &dagger;: alignment, *: Pg/Sg based; GUI-based otherwise</H2><br>\n\n'
    return string

def write_html(catalog, fnam_out, magnitude_version):
    output = create_html_header()
    output += catalog.get_event_count_table()
    output += add_information()
    output += create_table_head(
        column_names=(' ',
                      'name',
                      'type',
                      'LQ',
                      'Origin time<br>(UTC)',
                      'Start time<br>(UTC)',
                      'Start time<br>(LMST)',
                      'duration<br>[minutes]',
                      'distance<br>[degree]',
                      'SNR',
                      'P-amp<br>[m]',
                      'S-amp<br>[m]',
                      '2.4 Hz<br>pick [m]',
                      '2.4 Hz<br>fit [m]',
                      'A0<sup>2</sup><br>[dB]',
                      'MbP',
                      'MbS',
                      'M2.4',
                      'MFB',
                      'f_c<br>[Hz]',
                      'tstar<br>[s]',
                      'VBB<br>rate',
                      '100sps<br> SP1',
                      '100sps<br> SPH'))
    output_error = create_table_head(
        table_head='Events with errors',
        column_names=(' ',
                      'name',
                      'type',
                      'LQ',
                      'missing picks',
                      'picks in wrong order'))
    formats = ('%d', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s',
               '%s', '%8.2E', '%8.2E', '%8.2E', '%8.2E',
               '%4d&plusmn;%d',
               '%3.1f', '%3.1f',
               '%3.1f', '%3.1f&plusmn;%3.1f',
               '%3.1f', '%5.3f',
               '%s', '%s', '%s')
    time_string = {'GUI': '%s<sup>[O]</sup>',
                   'aligned': '%s<sup>[A]</sup>',
                   'PgSg': '%s',
                   'unknown': '%s'}
    dist_string = {'GUI': '{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}',
                   'aligned': '<i>{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}</i>&dagger;',
                   'PgSg': '<i>{0.distance:.3g}&plusmn;{0.distance_sigma:.2g}</i>*',
                   'unknown': '<i>-</i>'}
    event_type_idx = {'LF': 1,
                      'BB': 2,
                      'HF': 3,
                      '24': 4,
                      'VF': 5,
                      'SF': 6}
    ievent = len(catalog)
    print('Filling HTML table with event entries')
    error_events = False
    for event in tqdm(catalog):
        picks_check = check_picks(ievent, event)
        if picks_check is not None:
            row = '<tr> ' + picks_check + '</tr>\n'
            output_error += row
            error_events = True

        else:
            try:
                row = create_event_row(dist_string,
                                       time_string,
                                       event,
                                       event_type_idx,
                                       formats,
                                       ievent,
                                       magnitude_version=magnitude_version)
            except KeyError as e:
                print('Problem with event %s (%s-%s):' %
                      (event.name, event.mars_event_type_short, event.quality))

                print(e)
                print(event.picks)
                print(event.amplitudes)
                raise e
            else:
                output += row
        ievent -= 1
    footer = create_footer()
    output += footer
    if error_events:
        output_error += 4 * ' ' + '</tbody>\n </table>'
        output += output_error
    with open(fnam_out, 'w') as f:
        f.write(output)


def check_picks(ievent, event):
    missing_picks = []
    wrong_pairs = ''
    mandatory_minimum = ['start', 'end', 'noise_start', 'noise_end']

    for pick in mandatory_minimum:
        if pick not in event.picks or event.picks[pick] == '':
            missing_picks.append(pick)

    mandatory_ABC = ['P_spectral_start', 'P_spectral_end']
    if event.quality in ['A', 'B', 'C']:
        for pick in mandatory_ABC:
            if pick not in event.picks or event.picks[pick] == '':
                missing_picks.append(pick)

    pairs = [['P_spectral_start', 'P_spectral_end'],
             ['noise_start', 'noise_end'],
             ['start', 'end']]
    for pair in pairs:
        if not (event.picks[pair[0]] == '' or event.picks[pair[1]] == ''):
            if utct(event.picks[pair[0]]) > utct(event.picks[pair[0]]):
                print('Wrong order of picks' + pair)
                wrong_pairs += pair[0] + ', '

    if len(missing_picks) > 0:
        output = '<td>%d</td>\n <td>%s</td>\n' % (ievent, event.name)
        output += '<td>%s</td>\n <td>%s</td>\n' % (event.mars_event_type_short, event.quality)
        output += '<td> '
        for pick in missing_picks:
            output += pick + ', '
        output += '</td>\n <td>'
        if len(wrong_pairs) > 0:
            output += + wrong_pairs
        output += '</td>\n'
        return output
    else:
        False

def create_event_row(dist_string, time_string, event, event_type_idx, formats,
                     ievent,
                     magnitude_version='Giardini2020',
                     path_images_local='/usr/share/nginx/html/InSight_plots',
                     path_images='http://mars.ethz.ch/InSight_plots'):

    if event.origin_time == '':
        origin_time = '-'
    else:
        origin_time = event.origin_time.strftime('%Y-%m-%d<br>%H:%M:%S')

    utc_time = event.starttime.strftime('%Y-%m-%d<br>%H:%M:%S')
    lmst_time = solify(event.starttime).strftime('%H:%M:%S')
    duration = event.duration.strftime('%M:%S')
    event.fnam_report['name'] = event.name
    event.fnam_report['summary_local'] = pjoin(path_images_local,
                                               'event_summary',
                                               '%s_event_summary.png' %
                                               event.name)
    event.fnam_report['summary'] = pjoin(path_images,
                                         'event_summary',
                                         '%s_event_summary.png' %
                                         event.name)
    event.fnam_report['pol_local'] = pjoin(path_images_local,
                                           'event_plots',
                                           event.name,
                                           '%s_polarization.png' %
                                           event.name)
    event.fnam_report['pol'] = pjoin(path_images,
                                     'event_plots',
                                     event.name,
                                     '%s_polarization.png' %
                                     event.name)
    event.fnam_report['fb_local'] = pjoin(path_images_local,
                                          'filterbanks',
                                          event.mars_event_type_short,
                                          'filterbank_%s_all.png' %
                                          event.name)
    event.fnam_report['fb'] = pjoin(path_images,
                                    'filterbanks',
                                    event.mars_event_type_short,
                                    'filterbank_%s_all.png' %
                                    event.name)
    path_dailyspec = pjoin(path_images,
                           'spectrograms/by_channels/02.BHZ/',
                           'Sol%04d.Spectrogram_LF-02.BHZ__HF-02.BHZ.png'
                           % int(float(solify(event.starttime)) / 86400 + 1))
    try:
        if event.mars_event_type_short in ('HF', 'VF', '24'):
            snr = calc_stalta(event, fmin=2.2, fmax=2.8)
            snr_string = '%.1f (2.4Hz)' % snr
        elif event.mars_event_type_short == ('SF'):
            snr, snr_win = calc_SNR(event, fmin=8.0, fmax=12.,
                                    SP=True, hor=True)
            snr_string = '%.1f (%s, 8-12Hz)' % (snr, snr_win)
        else:
            snr, snr_win = calc_SNR(event, fmin=0.2, fmax=0.5)
            snr_string = '%.1f (%s, 2-5s)' % (snr, snr_win)

        sortkey = (ievent,
                   None,
                   event_type_idx[event.mars_event_type_short],
                   None,
                   float(utct(event.starttime)),
                   float(utct(event.starttime)),
                   float(solify(event.starttime)) % 86400,
                   None,
                   event.distance,
                   snr,
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
        if pexists(event.fnam_report['summary_local']):
            link_report = \
                ('<a href="{summary:s}" target="_blank">{name:s}</a><br>' +
                 '<a href="{Z:s}" target="_blank">Z</a> ' +
                 '<a href="{N:s}" target="_blank">N</a> ' +
                 '<a href="{E:s}" target="_blank">E</a>').format(
                    **event.fnam_report)
        else:
            link_report = \
                ('{name:s}<br>' +
                 '<a href="{Z:s}.html" target="_blank">Z</a> ' +
                 '<a href="{N:s}.html" target="_blank">N</a> ' +
                 '<a href="{E:s}.html" target="_blank">E</a>').format(
                    **event.fnam_report)
        if pexists(event.fnam_report['pol_local']):
            link_report += ' <a href="{pol:s}" target="_blank">Pol</a>'.format(
                **event.fnam_report)

        link_duration = '<a href="%s" target="_blank">%s</a>' % (
            event.fnam_report['fb'], duration)

        link_lmst = '<a href="%s" target="_blank">%s</a>' % (
            path_dailyspec, lmst_time)

        row = create_row(
            (ievent,
             link_report,
             event.mars_event_type_short,
             event.quality,
             time_string[event.distance_type] % origin_time,
             utc_time,
             link_lmst,
             link_duration,
             dist_string[event.distance_type].format(event),
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
             (event.amplitudes['A0']
              if event.amplitudes['A0'] is not None else None,
              event.amplitudes['A0_err']
              if event.amplitudes['A0_err'] is not None else None),
             event.magnitude(mag_type='mb_P', version=magnitude_version)[0],
             event.magnitude(mag_type='mb_S', version=magnitude_version)[0],
             event.magnitude(mag_type='m2.4', version=magnitude_version)[0],
             event.magnitude(mag_type='MFB', version=magnitude_version),
             event.amplitudes['f_c'],
             event.amplitudes['tstar'],
             event.available_sampling_rates()['VBB_Z'],
             _fmt_bool(event.available_sampling_rates()['SP_Z'] == 100.),
             _fmt_bool(event.available_sampling_rates()['SP_N'] == 100.),
             ),
            extras=sortkey,
            fmts=formats)

    except ValueError: # KeyError: #, AttributeError) as e:
        link_lmst = '<a href="%s" target="_blank">%s</a>' % (
            path_dailyspec, lmst_time)
        sortkey = (ievent,
                   None,
                   event_type_idx[event.mars_event_type_short],
                   None,
                   float(utct(event.picks['start'])),
                   float(solify(event.picks['start'])) % 86400,
                   0.)
        row = create_row((  # ievent, event.name, 'PRELIMINARY LOCATION'
            ievent,
            event.name,
            event.mars_event_type_short,
            event.quality,
            utc_time,
            link_lmst,
            'PRELIM'
            ),
            extras=sortkey)
    return row


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


def create_table_head(column_names, table_head='Event table'):
    output = ''
    output += '<article>\n'
    output += '  <header>\n'
    output += '    <h1>' + table_head + '</h1>\n'
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
                        nargs='+', default=('A', 'B', 'C', 'D'))

    helptext = 'Distances to use: "all" (default), "aligned", "GUI"'
    parser.add_argument('-d', '--distances', help=helptext,
                        default='all')

    helptext = 'Magnitude version to use: "Giardini2020" (default), "Boese2021"'
    parser.add_argument('-m', '--mag_version', help=helptext,
                        default='Giardini2020')

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
    if args.distances == 'all':
        catalog.load_distances(fnam_csv=args.input_dist)
        fnam_out='overview.html'
    elif args.distances == 'GUI':
        fnam_out='overview_GUI.html'
    elif args.distances == 'aligned':
        catalog.load_distances(fnam_csv=args.input_dist, overwrite=True)
        fnam_out='overview_aligned.html'

    inv = obspy.read_inventory(args.inventory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Read waveforms')
        catalog.read_waveforms(inv=inv, kind='DISP', sc3dir=args.sc3_dir)
    print('Calc spectra')
    catalog.calc_spectra(winlen_sec=20., detick_nfsamp=10)

    print('Plot filter banks')
    catalog.plot_filterbanks(dir_out='filterbanks', annotations=ann)

    print('Make magnitude reports')
    catalog.make_report(dir_out='reports', annotations=ann)

    print('Create table')
    catalog.write_table(fnam_out=fnam_out, magnitude_version=args.mag_version)
