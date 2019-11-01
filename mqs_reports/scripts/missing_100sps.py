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
    from obspy import UTCDateTime as utct

    args = define_arguments()
    catalog = Catalog(fnam_quakeml=args.input_quakeml,
                      type_select=args.types, quality=args.quality)
    ann = Annotations(fnam_csv=args.input_csv)
    inv = obspy.read_inventory(args.inventory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalog.read_waveforms(inv=inv, kind='DISP', sc3dir=args.sc3_dir)

    for event in catalog:

        av_data = event.available_sampling_rates()
        if av_data['SP_Z'] is None:
            print(event.name,
                  utct(event.starttime - 300).strftime('%Y-%m-%d %H:%M:%S'),
                  utct(event.endtime + 1200).strftime('%Y-%m-%d %H:%M:%S'))
