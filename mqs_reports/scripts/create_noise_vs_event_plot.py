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
from obspy import UTCDateTime as utct

from mqs_reports.catalog import Catalog
from mqs_reports.noise import Noise, read_noise


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

    return parser.parse_args()


args = define_arguments()

inv = obspy.read_inventory(args.inventory)
cat = Catalog(fnam_quakeml=args.input_quakeml,
              quality=['A', 'B', 'C', 'D'])
cat = cat.select(starttime='20190301',
              event_type=['HF', '24', 'LF', 'BB'])
noise = Noise(sc3_dir=args.sc3_dir,
              starttime=utct('20190202'),
              endtime=utct(),
              inv=inv,
              winlen_sec=120.
              )
noise.save('noise_from_20190202.npz')
# noise = read_noise('noise_from_20190202.npz')
# noise.read_quantiles('noise_quantiles.npz')
cat.load_distances(fnam_csv=args.input_dist)
cat.read_waveforms(inv=inv, sc3dir=args.sc3_dir)
cat.calc_spectra(winlen_sec=10.)
noise.plot_daystats(cat)
