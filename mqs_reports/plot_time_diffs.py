#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import matplotlib.pyplot as plt
import numpy as np

from mqs_reports.catalog import Catalog

# fig = plt.figure()
# events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191004.xml',
#                  type_select='higher')
# events.plot_pickdiffs('Sg', 'Pg', 'end', 'start', fig=fig, c='r', label='HF')
# events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191004.xml',
#                  type_select='2.4_HZ')
# events.plot_pickdiffs('Sg', 'Pg', 'end', 'start', fig=fig, c='b', label='2.4 '
#                                                                         'Hz')
# ax = fig.gca()
# ax.legend()
# plt.show()

fig = plt.figure()

vY = 1. / (1./2.0 - 1./2.0/np.sqrt(3.))
events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191004.xml',
                 type_select='higher')
events.plot_pickdiff_over_time('Sg', 'Pg', fig=fig, label='HF', vY=vY)
events = Catalog(fnam_quakeml='./mqs_reports/data/catalog_20191004.xml',
                 type_select='2.4_HZ')
events.plot_pickdiff_over_time('Sg', 'Pg', fig=fig, label='2.4 Hz', vY=vY)


ax = fig.gca()
ax.legend()
plt.show()
