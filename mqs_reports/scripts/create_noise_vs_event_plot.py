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

    helptext = 'Use old noise'
    parser.add_argument('--old_noise', default=False, help=helptext,
                        action='store_true')

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

inv = obspy.read_inventory(args.inventory)
if args.old_noise:
    noise = read_noise('noise.npz')
    #noise.read_quantiles('noise_quantiles.npz')
else:
    noise = Noise(sc3_dir=args.sc3_dir,
                  starttime=utct('20190202'),
                  endtime=utct(),
                  inv=inv,
                  winlen_sec=120.
                  )
    noise.save('./noise.npz')

extra = np.loadtxt('dds.txt', skiprows=1)
tau = np.loadtxt('mqs_reports/data/nsyt_tau_report.txt', skiprows=1,
                 usecols=[1, 2])
cat = Catalog(fnam_quakeml=args.input_quakeml,
              quality=['A', 'B', 'C', 'D'])
cat = cat.select(event_type=['VF', 'HF', 'LF', 'BB', '24'],
                 endtime='2019-06-01')
cat.load_distances(fnam_csv=args.input_dist)
cat.read_waveforms(inv=inv, sc3dir=args.sc3_dir)
cat.calc_spectra(winlen_sec=10.)
sol_end = args.sol_end
sol_start = args.sol_start

noise.plot_daystats(cat, data_apss=False,
                    sol_start=sol_start,
                    sol_end=sol_end,
                    grouping=args.grouping,
                    fnam_out='noise_vs_events.pdf')
noise.read_quantiles(fnam=f'noise_{args.grouping}.npz')
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

# noise.plot_daystats(cat, data_apss=True, extra_data=[extra[:, 0], extra[:,
# # 2]],#
#                     tau_data=[tau[:, 0], tau[:, 1]],#
#                     sol_start=sol_start,#
#                     sol_end=sol_end,#
#                     grouping='timewindows',#
#                     fnam_out='./noise_metal.png', metal=True)
