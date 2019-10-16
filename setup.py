#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Some python tools for the InSight mars mission.

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2018
    Martin van Driel (Martin@vanDriel.de)
:license:
    None
'''
from setuptools import setup, find_packages

setup(name='mqs_reports',
      version='0.1',
      description='Some python tools to create reports from the MQS database.',
      url='https://github.com/sstaehler/mqs_reports',
      author='Simon Stähler, Martin van Driel',
      author_email='staehler@erdw.ethz.ch',
      license='None',
      packages=find_packages(),
      install_requires=['obspy', 'plotly', 'lxml', 'numpy', 'matplotlib',
                        'tqdm', 'scipy'])
