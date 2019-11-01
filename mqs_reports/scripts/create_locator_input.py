#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from argparse import ArgumentParser

from mqs_reports.catalog import Catalog


def define_arguments():
    helptext = 'Create HTML overview table and individual event plots'
    parser = ArgumentParser(description=helptext)

    helptext = 'Input QuakeML BED file'
    parser.add_argument('input_quakeml', help=helptext)

    return parser.parse_args()


if __name__ == '__main__':
    args = define_arguments()
    catalog = Catalog(fnam_quakeml=args.input_quakeml,
                      quality=['A', 'B'])
    for event in catalog.select(event_type=['LF', 'BB']).events:
        print(event)
        event.write_locator_yaml(fnam_out=f'locator_in_{event.name}.yaml')
