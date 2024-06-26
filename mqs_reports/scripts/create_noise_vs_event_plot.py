#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''
from argparse import ArgumentParser

import numpy as np
import obspy

from mqs_reports.catalog import Catalog
from mqs_reports.noise import Noise, read_noise
from mqs_reports.utils import UTCify


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

    helptext = 'Use old noise'
    parser.add_argument('--old_noise', default=False, help=helptext,
                        action='store_true')

    helptext = 'Chose color scheme (standard, Knapmeyer)'
    parser.add_argument('--color_scheme', default='standard',
                        help=helptext, type=str)

    helptext = 'Use quantiles for grouping instead of time windows'
    parser.add_argument('--grouping', help=helptext, default='quantiles',
                        type=str)

    helptext = 'Start Sol'
    parser.add_argument('--sol_start', help=helptext,
                        default=80, type=int)

    helptext = 'End Sol'
    parser.add_argument('--sol_end', help=helptext,
                        default=550, type=int)

    return parser.parse_args()


args = define_arguments()

sol_end = args.sol_end
sol_start = args.sol_start

inv = obspy.read_inventory(args.inventory)
if args.old_noise:
    noise = read_noise('noise.npz')
else:
    noise = Noise(sc3_dir=args.sc3_dir,
                  starttime=UTCify((sol_start-1) * 86400),
                  endtime=UTCify(sol_end * 86400),
                  inv=inv,
                  winlen_sec=120.
                  )
    noise.save('./noise.npz')
noise.save_ascii('./noise.csv')

extra = np.loadtxt('./data/dds.txt', skiprows=1)
tau = np.loadtxt('./data/nsyt_tau_report.txt', skiprows=1,
                 usecols=[1, 2])
cat = Catalog(fnam_quakeml=args.input_quakeml,
              quality=['A', 'B', 'C', 'D'])
cat = cat.select(event_type=['VF', 'HF', 'LF', 'BB', '24'],
                 starttime=UTCify((sol_start-1) * 86400),
                 endtime=UTCify(sol_end * 86400))
cat.load_distances(fnam_csv=args.input_dist)
cat.read_waveforms(inv=inv, sc3dir=args.sc3_dir)
cat.calc_spectra(winlen_sec=10.)

noise.plot_daystats(cat, data_apss=False,
                    sol_start=sol_start,
                    sol_end=sol_end,
                    grouping=args.grouping,
                    fnam_out='noise_vs_events.png')
noise.read_quantiles(fnam=f'noise_{args.grouping}.npz')
noise.plot_daystats(cat, data_apss=False,
                    sol_start=sol_start,
                    sol_end=sol_end,
                    grouping=args.grouping,
                    fnam_out='noise_vs_events.pdf')
noise.plot_daystats(cat, data_apss=True, extra_data=[extra[:, 0], extra[:, 2]],
                    tau_data=[tau[:, 0], tau[:, 1]],
                    sol_start=sol_start,
                    sol_end=sol_end,
                    grouping=args.grouping,
                    fnam_out='./noise_apss.png', metal=False)
noise.plot_daystats(cat, data_apss=True, extra_data=[extra[:, 0], extra[:, 2]],
                    tau_data=[tau[:, 0], tau[:, 1]],
                    sol_start=sol_start,
                    sol_end=sol_end,
                    grouping=args.grouping,
                    fnam_out='./noise_apss.pdf', metal=False)
noise.plot_daystats(cat, data_apss=False,
                    sol_start=sol_start,
                    sol_end=567,
                    grouping=args.grouping,
                    color_scheme='Knapmeyer',
                    fnam_out='noise_vs_events_knapmeyer.pdf')

# noise.plot_daystats(cat, data_apss=True, extra_data=[extra[:, 0], extra[:,
# # 2]],#
#                     tau_data=[tau[:, 0], tau[:, 1]],#
#                     sol_start=sol_start,#
#                     sol_end=sol_end,#
#                     grouping='timewindows',#
#                     fnam_out='./noise_metal.png', metal=True)
