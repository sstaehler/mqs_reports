#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

from glob import glob
from os import makedirs
from os.path import join as pjoin

import numpy as np
import obspy
from obspy import UTCDateTime as utct
from obspy.geodetics.base import locations2degrees

from mqs_reports.magnitudes import fit_spectra
from mqs_reports.utils import create_fnam_event, read_data, calc_PSD

LANDER_LAT = 4.5024
LANDER_LON = 135.6234

class Event:
    def __init__(self,
                 name: str,
                 publicid: str,
                 origin_publicid: str,
                 picks: dict,
                 quality: str,
                 latitude: float,
                 longitude: float,
                 mars_event_type: str):
        self.name = name.strip()
        self.publicid = publicid
        self.origin_publicid = origin_publicid
        self.picks = picks
        self.quality = quality[-1]
        self.mars_event_type = mars_event_type.split('#')[-1]

        # Set a short event type
        if self.mars_event_type == 'HIGH_FREQUENCY':
            self.mars_event_type_short = 'HF'
        elif self.mars_event_type == 'VERY_HIGH_FREQUENCY':
            self.mars_event_type_short = 'VF'
        elif self.mars_event_type == 'BROADBAND':
            self.mars_event_type_short = 'BB'
        elif self.mars_event_type == 'LOW_FREQUENCY':
            self.mars_event_type_short = 'LF'
        elif self.mars_event_type == '2.4_HZ':
            self.mars_event_type_short = '24'
        self.amplitudes = dict()

        # Set distance or calculate it for HF, VHF and 2.4 events
        self.latitude = latitude
        self.longitude = longitude
        self.distance_type = ''
        if (abs(self.latitude - LANDER_LAT) > 1e-3 and
                abs(self.longitude - LANDER_LON) > 1e-3):
            self.distance = locations2degrees(lat1=self.latitude,
                                              long1=self.longitude,
                                              lat2=LANDER_LAT,
                                              long2=LANDER_LON)
            self.distance_type = 'GUI'
        elif self.mars_event_type_short in ['HF', 'VF', '24']:
            self.distance = self.calc_distance()
            if self.distance is not None:
                self.distance_type = 'PgSg'
        else:
            self.distance = None
        self._waveforms_read = False
        self._spectra_available = False

    def calc_distance(self, vp=np.sqrt(3) * 2.0, vs=2.0):
        if len(self.picks['Sg']) > 0 and len(self.picks['Pg']) > 0:
            deltat = float(utct(self.picks['Sg']) - utct(self.picks['Pg']))
            return deltat / (1. / vs - 1. / vp) / 55.
        else:
            return None

    def read_waveforms(self, inv, kind, sc3dir
                       ):
        if not self.read_data_local():
            self.read_data_from_sc3dir(inv, kind, sc3dir)
            self.write_data_local()
        self._waveforms_read = True


    def read_data_local(self):
        event_path = pjoin('events', '%s' % self.name)
        waveform_path = pjoin(event_path, 'waveforms')
        origin_path = pjoin(event_path, 'origin_id.txt')
        success = False
        if len(glob(origin_path)) > 0:
            with open(origin_path, 'r') as f:
                origin_local = f.readline().strip()
            if origin_local == self.origin_publicid:
                try:
                    self.waveforms_VBB = obspy.read(pjoin(waveform_path,
                                                          'waveforms_VBB.mseed'))
                    success = True
                except(TypeError):
                    success = False
                SP_path = pjoin(waveform_path, 'waveforms_SP.mseed')
                if len(glob(SP_path)):
                    self.waveforms_SP = obspy.read(SP_path)
                else:
                    self.waveforms_SP = None
        return success


    def write_data_local(self):
        event_path = pjoin('events', '%s' % self.name)
        waveform_path = pjoin(event_path, 'waveforms')
        origin_path = pjoin(event_path, 'origin_id.txt')
        makedirs(waveform_path, exist_ok=True)

        with open(origin_path, 'w') as f:
            f.write(self.origin_publicid)
        self.waveforms_VBB.write(pjoin(waveform_path,
                                       'waveforms_VBB.mseed'),
                                 format='MSEED', encoding='FLOAT64')
        if self.waveforms_SP is not None and len(self.waveforms_SP) > 0:
            self.waveforms_SP.write(pjoin(waveform_path,
                                          'waveforms_SP.mseed'),
                                    format='MSEED', encoding='FLOAT64')

    def read_data_from_sc3dir(self,
                              inv, kind, sc3dir, tpre_SP=100, tpre_VBB=900.):
        self.kind = kind

        if len(self.picks['noise_start']) > 0:
            twin_start = min((utct(self.picks['start']),
                              utct(self.picks['noise_start'])))
        else:
            twin_start = utct(self.picks['start'])
        if len(self.picks['noise_end']) > 0:
            twin_end = max((utct(self.picks['end']),
                            utct(self.picks['noise_end'])))
        else:
            twin_end = utct(self.picks['end'])

        filenam_SP_HG = 'XB.ELYSE.65.EH?.D.2019.%03d'
        fnam_SP = create_fnam_event(
            filenam_inst=filenam_SP_HG,
            sc3dir=sc3dir, time=self.picks['start'])

        if len(glob(fnam_SP)) > 0:
            # Use SP waveforms only if 65.EH? exists, not otherwise (we
            # don't need 20sps SP data)
            self.waveforms_SP = read_data(fnam_SP, inv=inv, kind=kind,
                                             twin=[twin_start - tpre_SP,
                                                    twin_end + tpre_SP],
                                             fmin=0.5)
        else:
            self.waveforms_SP = None

        # Try for 02.BH? (20sps VBB)
        success_VBB = False
        filenam_VBB_HG = 'XB.ELYSE.02.BH?.D.2019.%03d'
        fnam_VBB = create_fnam_event(
            filenam_inst=filenam_VBB_HG,
            sc3dir=sc3dir, time=self.picks['start'])
        if len(glob(fnam_VBB)) == 3:
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True

        if not success_VBB:
            # Try for 03.BH? (10sps VBB)
            filenam_VBB_HG = 'XB.ELYSE.03.BH?.D.2019.%03d'
            fnam_VBB = create_fnam_event(
                filenam_inst=filenam_VBB_HG,
                sc3dir=sc3dir, time=self.picks['start'])
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True

        if not success_VBB:
            self.waveforms_VBB = None


    def calc_spectra(self, winlen_sec):
        if not self._waveforms_read:
            raise RuntimeError('waveforms not read in Event object\n' +
                               'Call Event.read_waveforms() first.')
        twins = (((self.picks['start']),
                  (self.picks['end'])),
                 ((self.picks['noise_start']),
                  (self.picks['noise_end'])),
                 ((self.picks['P_spectral_start']),
                  (self.picks['P_spectral_end'])),
                 ((self.picks['S_spectral_start']),
                  (self.picks['S_spectral_end'])))
        self.spectra = dict()
        self.spectra_SP = dict()
        variables = ('all',
                     'noise',
                     'P',
                     'S')
        for twin, variable in zip(twins, variables):
            self.spectra[variable] = dict()
            if len(twin[0]) == 0:
                self.spectra[variable] = None
                continue
            for chan in ['Z', 'N', 'E']:
                # f, p = read_spectrum(fnam_base=fnam_spectrum,
                #                      variable=variable,
                #                      chan=chan,
                #                      origin_publicid=self[
                #                          'origin_publicid'])
                # if f is None:
                st_sel = self.waveforms_VBB.select(
                    channel='??' + chan)
                if len(st_sel) > 0:
                    tr = st_sel[0].slice(starttime=utct(twin[0]),
                                         endtime=utct(twin[1]))
                    if tr.stats.npts > 0:
                        f, p = calc_PSD(tr,
                                        winlen_sec=winlen_sec)
                        self.spectra[variable]['p_' + chan] = p
                    else:
                        f = np.arange(0, 1, 0.1)
                        p = np.zeros((10))
                self.spectra[variable]['f'] = f

            if self.waveforms_SP is not None:
                self.spectra_SP[variable] = dict()
                for chan in ['Z', 'N', 'E']:
                    st_sel = self.waveforms_SP.select(
                        channel='??' + chan)
                    if len(st_sel) > 0:
                        tr = st_sel[0].slice(starttime=utct(twin[0]),
                                             endtime=utct(twin[1]))
                        if tr.stats.npts > 0:
                            f, p = calc_PSD(tr,
                                            winlen_sec=winlen_sec)
                            self.spectra_SP[variable]['p_' + chan] = p
                        else:
                            f = np.arange(0, 1, 0.1)
                            p = np.zeros((10))
                            self.spectra_SP[variable]['p_' + chan] = p
                            self.spectra_SP[variable]['f_' + chan] = f
                    else:
                        # Case that only SP1==SPZ is switched on
                        self.spectra_SP[variable]['p_' + chan] = \
                            np.zeros_like(p)
                self.spectra_SP[variable]['f'] = f

        # try:
        if self.spectra['S'] is not None:
            sig_spec = self.spectra['S']['p_Z']
        elif 'noise' in self.spectra and 'all' in self.spectra:
            sig_spec = self.spectra['all']['p_Z']
        self.amplitudes = \
            fit_spectra(self.spectra['noise']['f'],
                        sig_spec,
                        self.spectra['noise']['p_Z'],
                        type=self.mars_event_type_short)
        # except KeyError:
        #     print('Some time windows missing for event %s' % self.name)
        #     print(self.spectra)

        self._spectra_available = True

    def pick_amplitude(self, pick, comp, fmin, fmax, instrument='VBB',
                       unit=None):
        if not self._waveforms_read:
            raise RuntimeError('waveforms not read in Event object\n' +
                               'Call Event.read_waveforms() first.')

        if instrument=='VBB':
            st_work = self.waveforms_VBB.copy()
        else:
            st_work = self.waveforms_SP.copy()

        st_work.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)

        if unit is 'nm':
            output_fac = 1e9
        elif unit is 'pm':
            output_fac = 1e12
        elif unit is 'fm':
            output_fac = 1e15
        else:
            output_fac = 1.
        # if not self.kind=='DISP':
        #     raise ValueError('Waveform must be displacement for amplitudes')
        if self.picks[pick]=='':
            return None
        else:
            tmin = utct(self.picks[pick]) - 10.
            tmax = utct(self.picks[pick]) + 10.
            st_work.trim(starttime=tmin, endtime=tmax)
            if comp in ['Z', 'N', 'E']:
                return abs(st_work.select(channel='??' + comp)[0].data).max() \
                       * output_fac
            elif comp == 'all':
                amp_N = abs(st_work.select(channel='??N')[0].data).max()
                amp_E = abs(st_work.select(channel='??E')[0].data).max()
                amp_Z = abs(st_work.select(channel='??Z')[0].data).max()
                return max((amp_E, amp_N, amp_Z)) * output_fac
            elif comp == 'horizontal':
                amp_N = abs(st_work.select(channel='??N')[0].data).max()
                amp_E = abs(st_work.select(channel='??E')[0].data).max()
                return max((amp_E, amp_N)) * output_fac
            elif comp == 'vertical':
                return abs(st_work.select(channel='??Z')[0].data).max() \
                       * output_fac

    def magnitude(self, type, instrument='VBB', distance=None):
        import mqs_reports.magnitudes as mag
        pick_name = {'mb_P': 'Peak_MbP',
                     'mb_S': 'Peak_MbS',
                     'm2.4': 'Peak_M2.4',
                     'MFB': None
                     }
        freqs = {'mb_P': (1. / 6., 1. / 2.),
                 'mb_S': (1. / 6., 1. / 2.),
                 'm2.4': (2., 3.),
                 'MFB': None
                 }
        component = {'mb_P': 'vertical',
                     'mb_S': 'horizontal',
                     'm2.4': 'vertical',
                     'MFB': None
                     }
        funcs = {'mb_P': mag.mb_P,
                 'mb_S': mag.mb_S,
                 'm2.4': mag.M2_4,
                 'MFB': mag.MFB
                 }
        if self.distance is None and distance is None:
            return None
        elif self.distance is not None:
            distance = self.distance
        if type in ('mb_P', 'mb_S'):  #, 'm2.4'):
            amplitude = self.pick_amplitude(pick=pick_name[type],
                                            comp=component[type],
                                            fmin=freqs[type][0],
                                            fmax=freqs[type][1],
                                            instrument=instrument
                                            )
        elif type == 'MFB':
            amplitude = self.amplitudes['A0']
        elif type == 'm2.4':
            amplitude = self.amplitudes['A_24']

        else:
            raise ValueError('unknown magnitude type %s' % type)

        if amplitude is None:
            return None
        else:
            return funcs[type](amplitude=amplitude,
                               distance=distance)

    def make_report(self, fnam_out):
        from mqs_reports.report import make_report
        make_report(self, fnam_out)
