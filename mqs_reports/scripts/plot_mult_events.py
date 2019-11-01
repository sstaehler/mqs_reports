#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime as utct


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

    catalog = catalog.select(name=['S0173a', 'S0235b', 'S0325a'])
    # catalog.select(name='S0262b')
    ann = Annotations(fnam_csv=args.input_csv)
    # load manual (aligned) distances
    catalog.load_distances(fnam_csv=args.input_dist)
    inv = obspy.read_inventory(args.inventory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalog.read_waveforms(inv=inv, kind='DISP', sc3dir=args.sc3_dir)

    fig, ax = plt.subplots(3, 2)
    t_pre = 250.
    t_post = 350.

    for ichan, chan in enumerate(['BHZ', 'BHE', 'BHN']):
        for i, event in enumerate(catalog):
            tr = event.waveforms_VBB.select(channel=chan).copy()[0]
            tr.differentiate()
            tr.filter('highpass', freq=1. / 15, zerophase=True)
            tr.filter('lowpass', freq=1. / 7)
            tr.trim(starttime=utct(event.picks['S']) - t_pre,
                    endtime=utct(event.picks['S']) + t_post)
            x = tr.times() - t_pre
            y = tr.data / tr.data.max()
            ax[ichan][0].plot(x, y + i, label=event.name)

        for i, event in enumerate(catalog):
            tr = event.waveforms_VBB.select(channel=chan).copy()[0]
            tr.differentiate()
            tr.filter('highpass', freq=1. / 8)
            tr.filter('lowpass', freq=1. / 2)
            tr.trim(starttime=utct(event.picks['S']) - t_pre,
                    endtime=utct(event.picks['S']) + t_post)
            x = tr.times() - t_pre
            y = tr.data / tr.data.max()
            ax[ichan][1].plot(x, y + i, label=event.name)
        ax[ichan][0].legend()
    plt.show()
