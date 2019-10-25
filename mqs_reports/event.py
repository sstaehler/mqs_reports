#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

import inspect
from glob import glob
from os import makedirs
from os.path import join as pjoin
from typing import Union

import numpy as np
import obspy
from obspy import UTCDateTime as utct
from obspy.geodetics.base import locations2degrees, kilometers2degrees

from mqs_reports.magnitudes import fit_spectra
from mqs_reports.utils import create_fnam_event, read_data, calc_PSD

LANDER_LAT = 4.5024
LANDER_LON = 135.6234


EVENT_TYPES_SHORT = {
    'VERY_HIGH_FREQUENCY': 'VF',
    'BROADBAND': 'BB',
    'LOW_FREQUENCY': 'LF',
    'HIGH_FREQUENCY': 'HF',
    '2.4_HZ': '24'}

EVENT_TYPES = EVENT_TYPES_SHORT.keys()


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
        self.duration = utct(utct(self.picks['end']) -
                             utct(self.picks['start']))
        self.duration_s = utct(self.picks['end']) - utct(self.picks['start'])
        self.starttime = utct(utct(self.picks['start']))

        self.amplitudes = dict()

        # Set distance or calculate it for HF, VHF and 2.4 events
        self.latitude = latitude
        self.longitude = longitude
        self.distance_type = 'unknown'
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

        # Define Instance attributes
        self.waveforms_VBB = None
        self.waveforms_SP = None
        self.kind = None
        self.spectra = None
        self.spectra_SP = None

    @property
    def mars_event_type_short(self):
        return EVENT_TYPES_SHORT[self.mars_event_type]

    def __str__(self):
        if self.distance is not None:
            string = "Event {name} ({mars_event_type_short}-{quality}), " \
                     "distance: {distance:5.1f} degree ({distance_type})"
        else:
            string = "Event {name} ({mars_event_type_short}-{quality}), " \
                     "unknown distance"
        return string.format(**dict(inspect.getmembers(self)))

    def load_distance_manual(self,
                             fnam_csv: str) -> None:
        """
        Load distance of event from CSV file. Can be used for "aligned"
        distances that are not in the database
        :param: fnam_csv: path to CSV file with distances
        """
        from csv import DictReader
        with open(fnam_csv, 'r') as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                if self.distance is None:
                    if self.name == row['name']:
                        self.distance = float(row['distance'])
                        self.distance_type = 'aligned'

    def calc_distance(self,
                      vp: float = np.sqrt(3) * 2.0,
                      vs: float = 2.0) -> Union[float, None]:
        """
        Calculate distance of event based on Pg and Sg picks, if available,
        otherwise return None
        :param vp: P-velocity
        :param vs: S-velocity
        :return: distance in degree or None if no picks available
        """
        if len(self.picks['Sg']) > 0 and len(self.picks['Pg']) > 0:
            deltat = float(utct(self.picks['Sg']) - utct(self.picks['Pg']))
            distance_km = deltat / (1. / vs - 1. / vp)
            distance_deg = kilometers2degrees(distance_km, radius=2889)
        else:
            # TODO: replace to use the velocities from the arguments
            t0 = 600.
            d0 = 8.
            t1 = 1200
            d1 = 35.
            # fudge factor to compensate observation bias (low SNR events are
            # observed shorter)
            distance_deg = (d1 - d0) / (t1 - t0) * (self.duration_s * 1. - t0) \
                         + d0
            if distance_deg < 0:
                distance_deg = None
        return distance_deg

    def read_waveforms(self,
                       inv: obspy.Inventory,
                       sc3dir: str,
                       event_tmp_dir='./events',
                       kind: str = 'DISP') -> None:
        """
        Wrapper to check whether local copy of corrected waveform exists and
        read it from sc3dir otherwise (and create local copy)
        :param inv: Obspy.Inventory to use for instrument correction
        :param sc3dir: path to data, in SeisComp3 directory structure
        :param kind: 'DISP', 'VEL' or 'ACC'. Note that many other functions
                     expect the data to be in displacement
        """
        if not self.read_data_local(dir_cache=event_tmp_dir):
            self.read_data_from_sc3dir(inv, sc3dir, kind)
            self.write_data_local(dir_cache=event_tmp_dir)
        self._waveforms_read = True
        self.kind = 'DISP'

    def read_data_local(self, dir_cache: str = 'events') -> bool:
        """
        Read waveform data from local cache structure
        :param dir_cache: path to local cache
        :return: True if waveform was found in local cache
        """
        event_path = pjoin(dir_cache, '%s' % self.name)
        waveform_path = pjoin(event_path, 'waveforms')
        origin_path = pjoin(event_path, 'origin_id.txt')
        success = False
        if len(glob(origin_path)) > 0:
            with open(origin_path, 'r') as f:
                origin_local = f.readline().strip()
            if origin_local == self.origin_publicid:
                try:
                    self.waveforms_VBB = obspy.read(
                        pjoin(waveform_path, 'waveforms_VBB.mseed'))
                    success = True
                except TypeError:
                    success = False
                SP_path = pjoin(waveform_path, 'waveforms_SP.mseed')
                if len(glob(SP_path)):
                    self.waveforms_SP = obspy.read(SP_path)
                else:
                    self.waveforms_SP = None
        return success

    def write_data_local(self, dir_cache: str = 'events'):
        """
        Store waveform data in local cache structure
        @TODO: Save parameters (kind, filter) into file name
        :param dir_cache: path to local cache
        :return:
        """
        event_path = pjoin(dir_cache, '%s' % self.name)
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
                              inv: obspy.Inventory,
                              sc3dir: str,
                              kind: str,
                              tpre_SP: float = 100,
                              tpre_VBB: float = 900.) -> None:
        """
        Read waveform data into event object
        :param inv: obspy.Inventory object to use for instrument correction
        :param sc3dir: path to data, in SeisComp3 directory structure
        :param kind: Unit to correct waveform into ('DISP', 'VEL', 'ACC')
        :param tpre_SP: prefetch time for SP data (default: 100 sec)
        :param tpre_VBB: prefetch time for VBB data (default: 900 sec)
        """
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
        if len(glob(fnam_VBB)) % 3 == 0:
            try:
                self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                               kind=kind,
                                               twin=[twin_start - tpre_VBB,
                                                     twin_end + tpre_VBB])
            except Exception:
                success_VBB = False
            else:
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
        """
        Add spectra to event object.
        Spectra are stored in dictionaries
            event.spectra for VBB
            event.spectra_SP for SP
        Spectra are calculated separately for time windows "noise", "all",
        "P" and "S". If any of the necessary picks is missing, this entry is
        set to None.
        :param winlen_sec: window length for Welch estimator
        """
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
            spectrum_variable = dict()
            if len(twin[0]) == 0:
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
                tr = st_sel[0].slice(starttime=utct(twin[0]),
                                     endtime=utct(twin[1]))
                if tr.stats.npts > 0:
                    f, p = calc_PSD(tr,
                                    winlen_sec=winlen_sec)
                    spectrum_variable['p_' + chan] = p
                else:
                    f = np.arange(0, 1, 0.1)
                    p = np.zeros(10)
                spectrum_variable['f'] = f
            if len(spectrum_variable) > 0:
                self.spectra[variable] = spectrum_variable

            if self.waveforms_SP is not None:
                spectrum_variable = dict()
                for chan in ['Z', 'N', 'E']:
                    st_sel = self.waveforms_SP.select(
                        channel='??' + chan)
                    if len(st_sel) > 0:
                        tr = st_sel[0].slice(starttime=utct(twin[0]),
                                             endtime=utct(twin[1]))
                        if tr.stats.npts > 0:
                            f, p = calc_PSD(tr,
                                            winlen_sec=winlen_sec)
                            spectrum_variable['p_' + chan] = p
                        else:
                            f = np.arange(0, 1, 0.1)
                            p = np.zeros(10)
                            spectrum_variable['p_' + chan] = p
                            spectrum_variable['f_' + chan] = f
                    else:
                        # Case that only SP1==SPZ is switched on
                        spectrum_variable['p_' + chan] = \
                            np.zeros_like(p)
                spectrum_variable['f'] = f

            if len(spectrum_variable) > 0:
                self.spectra_SP[variable] = spectrum_variable

        self.amplitudes = {'A0': None,
                           'tstar': None,
                           'A_24': None,
                           'f_24': None,
                           'width_24': None}
        if 'noise' in self.spectra:
            f = self.spectra['noise']['f']
            p_noise = self.spectra['noise']['p_Z']
            for signal in ['S', 'P', 'all']:
                amplitudes = None
                if signal in self.spectra:
                    p_sig = self.spectra[signal]['p_Z']
                    amplitudes = fit_spectra(f=f,
                                             p_sig=p_sig,
                                             p_noise=p_noise,
                                             event_type=self.mars_event_type_short)
                if amplitudes is not None:
                    break
            if amplitudes is not None:
                self.amplitudes = amplitudes
            # if self.spectra['S'] is not None:
            #     sig_spec = self.spectra['S']['p_Z']
            # elif 'noise' in self.spectra and 'all' in self.spectra:
            #     sig_spec = self.spectra['all']['p_Z']
            # self.amplitudes = \
            #     fit_spectra(self.spectra['noise']['f'],
            #                 sig_spec,
            #                 self.spectra['noise']['p_Z'],
            #                 type=self.mars_event_type_short)
        # except KeyError:
        #     print('Some time windows missing for event %s' % self.name)
        #     print(self.spectra)

        self._spectra_available = True

    def pick_amplitude(self,
                       pick: str,
                       comp: str,
                       fmin: float,
                       fmax: float,
                       instrument: str = 'VBB',
                       unit: str = 'm') -> Union[float, None]:
        """
        Pick amplitude from waveform
        :param pick: name of pick to use. Corresponds to naming in the MQS
                     data model
        :param comp: component to pick on, can be 'E', 'N', 'Z' or
                     'horizontal', in which case maximum value along
                     horizontals is returned
        :param fmin: minimum frequency for pre-picking bandpass
        :param fmax: maximum frequency for pre-picking bandpass
        :param instrument: 'VBB' (default) or 'SP'
        :param unit: 'm', 'nm', 'pm', 'fm'
        :return: amplitude in 5 sec time window around pick time
        """
        if not self._waveforms_read:
            raise RuntimeError('waveforms not read in Event object\n' +
                               'Call Event.read_waveforms() first.')

        if instrument == 'VBB':
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
        elif unit is 'm':
            output_fac = 1.
        else:
            raise ValueError('Unknown unit %s' % unit)

        if not self.kind == 'DISP':
            raise RuntimeError('Waveform must be displacement for amplitudes')

        if self.picks[pick] == '':
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

    def magnitude(self,
                  mag_type: str,
                  distance: float = None,
                  instrument: str = 'VBB') -> Union[float, None]:
        """
        Calculate magnitude of an event
        :param mag_type: 'mb_P', 'mb_S' 'm2.4' or 'MFB':
        :param distance: float or None, in which case event.distance is used
        :param instrument: 'VBB' or 'SP'
        :return:
        """
        import mqs_reports.magnitudes as mag
        pick_name = {'mb_P': 'Peak_MbP',
                     'mb_S': 'Peak_MbS',
                     'm2.4': None,
                     'MFB': None
                     }
        freqs = {'mb_P': (1. / 6., 1. / 2.),
                 'mb_S': (1. / 6., 1. / 2.),
                 'm2.4': None,
                 'MFB': None
                 }
        component = {'mb_P': 'vertical',
                     'mb_S': 'horizontal',
                     'm2.4': None,
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
        if mag_type in ('mb_P', 'mb_S'):
            amplitude = self.pick_amplitude(pick=pick_name[mag_type],
                                            comp=component[mag_type],
                                            fmin=freqs[mag_type][0],
                                            fmax=freqs[mag_type][1],
                                            instrument=instrument
                                            )
            if amplitude is not None:
                amplitude = 20 * np.log10(amplitude)

        elif mag_type == 'MFB':
            amplitude = self.amplitudes['A0'] \
                if 'A0' in self.amplitudes else None
        elif mag_type == 'm2.4':
            amplitude = self.amplitudes['A_24'] \
                if 'A_24' in self.amplitudes else None

        else:
            raise ValueError('unknown magnitude type %s' % mag_type)

        if amplitude is None:
            return None
        else:
            return funcs[mag_type](amplitude_dB=amplitude,
                                   distance_degree=distance)

    def make_report(self, fnam_out, annotations=None):
        from mqs_reports.report import make_report
        make_report(self, fnam_out, annotations)
