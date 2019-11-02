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
from mqs_reports.catalog import Catalog
from mqs_reports.utils import autocorrelation
from obspy import UTCDateTime as utct


def define_arguments():
    helptext = 'Create Noise time evolution vs event amplitde overview plot'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

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


args = define_arguments()

inv = obspy.read_inventory(args.inventory)
cat = Catalog(fnam_quakeml=args.input_quakeml,
              type_select=args.types, quality=args.quality)
print(cat)
cat = cat.select(name=['S0260a', 'S0218a', 'S0264e', 'S0289a', 'S0308a',
                       'S0311a', 'S0128a'])
cat.read_waveforms(inv=inv, sc3dir=args.sc3_dir)
for event in cat:
    fig = autocorrelation(st=event.waveforms_VBB.copy(),
                          starttime=utct(event.picks['Pg']) - 10.,
                          endtime=utct(event.picks['Pg']) + 110.)
    fig.savefig('AC_%s_P.png' % event.name, dpi=200)
    fig = autocorrelation(st=event.waveforms_VBB.copy(),
                          starttime=utct(event.picks['Sg']) - 10.,
                          endtime=utct(event.picks['Sg']) + 230.)
    fig.savefig('AC_%s_S.png' % event.name, dpi=200)
