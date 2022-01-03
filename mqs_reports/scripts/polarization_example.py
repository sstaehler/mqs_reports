import obspy
from obspy import UTCDateTime as utct

from mqs_reports.catalog import Catalog

args = dict(input_quakeml='../../mqs_reports/mqs_reports/data/catalog.xml',
            inventory='../../mqs_reports/mqs_reports/data/inventory_single_epoch.xml',
            sc3_dir='../../mqs_reports/mnt/')



catalog = Catalog(fnam_quakeml=args['input_quakeml'],
                  type_select='lower', quality=['A', 'B'])
catalog = catalog.select(starttime='2021-12-24T00:00:00Z')

inv = obspy.read_inventory(args['inventory'])
for ev in catalog:
    ev.read_waveforms(inv=inv, kind='DISP', sc3dir=args['sc3_dir'], t_pad_VBB=3600.)
ev = catalog.events[0]

ev.plot_polarisation(t_pick_P=[10, 20], t_pick_S=[10, 20], alpha_elli=0., baz=45., alpha_inc=-1., #alpha_azi=1.,
                     vmax=-155,
                     fmin=0.02, fmax=3., dop_specwidth=1.2, dop_winlen=10.,
                     tstart=utct('2021-12-24T22:40:00Z'),
                     tend=utct('2021-12-25T01:34:00Z'),
                     show=True)