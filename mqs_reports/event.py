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
from obspy.geodetics.base import kilometers2degrees, gps2dist_azimuth

from mqs_reports.annotations import Annotations
from mqs_reports.magnitudes import fit_spectra
from mqs_reports.utils import create_fnam_event, read_data, calc_PSD, detick, \
    calc_cwf, solify

RADIUS_MARS = 3389.5
CRUST_VP = 4.
CRUST_VS = 4. / 3. ** 0.5
LANDER_LAT = 4.5024
LANDER_LON = 135.6234

EVENT_TYPES_SHORT = {
        'SUPER_HIGH_FREQUENCY': 'SF',
        'VERY_HIGH_FREQUENCY':  'VF',
    'BROADBAND': 'BB',
    'LOW_FREQUENCY': 'LF',
    'HIGH_FREQUENCY': 'HF',
    '2.4_HZ': '24'}

EVENT_TYPES_PRINT = {
    'SUPER_HIGH_FREQUENCY': 'super high frequency',
    'VERY_HIGH_FREQUENCY': 'very high frequency',
    'BROADBAND': 'broadband',
    'LOW_FREQUENCY': 'low frequency',
    'HIGH_FREQUENCY': 'high frequency',
    '2.4_HZ': '2.4 Hz'}

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
                 sso_distance: float,
                 sso_origin_time: str,
                 mars_event_type: str,
                 origin_time: str):
        self.name = name.strip()
        self.publicid = publicid
        self.origin_publicid = origin_publicid
        self.picks = picks
        self.quality = quality[-1]
        self.mars_event_type = mars_event_type.split('#')[-1]

        try:
            self.sol = solify(utct(self.picks['start'])).julday
            self.starttime = utct(utct(self.picks['start']))
            self.endtime = utct(utct(self.picks['end']))
            self.duration = utct(utct(self.picks['end']) -
                                 utct(self.picks['start']))
            self.duration_s = utct(self.picks['end']) - utct(
                self.picks['start'])
        except TypeError:
            print('incomplete picks for event %s' % self.name)
            print(self.picks)

        self.amplitudes = dict()

        # Set distance or calculate it for HF, VHF and 2.4 events
        self.latitude = latitude
        self.longitude = longitude
        self.distance_type = 'unknown'

        # Case that location was determined from BAZ and distance
        if (abs(self.latitude - LANDER_LAT) > 1e-3 and
                abs(self.longitude - LANDER_LON) > 1e-3):
            dist_km, az, baz = gps2dist_azimuth(lat1=self.latitude,
                                                lon1=self.longitude,
                                                lat2=LANDER_LAT,
                                                lon2=LANDER_LON,
                                                a=RADIUS_MARS)
            self.distance = kilometers2degrees(dist_km,
                                               radius=RADIUS_MARS)
            self.baz = baz
            self.az = az
            self.origin_time = utct(origin_time)
            self.distance_type = 'GUI'

        # Case that distance exists, but not BAZ. Then, distance and origin
        # time should be taken from SSO (ie the locator PDF output)
        elif sso_distance is not None:
            self.origin_time = utct(sso_origin_time)
            self.distance = sso_distance
            self.distance_type = 'GUI'
            self.baz = None

        # Case that distance can be estimated from Pg/Sg arrivals
        elif self.mars_event_type_short in ['HF', 'SF', 'VF', '24']:
            self.distance = self.calc_distance()
            if self.distance is not None:
                self.distance_type = 'PgSg'
            self.origin_time = utct(origin_time)
            self.baz = None

        else:
            self.origin_time = utct(origin_time)
            self.distance = None
            self.baz = None

        self._waveforms_read = False
        self._spectra_available = False

        # Define Instance attributes
        self.waveforms_VBB = None
        self.waveforms_SP = None
        self.kind = None
        self.spectra = None
        self.spectra_SP = None

        self.fnam_report = dict()

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
                             fnam_csv: str,
                             overwrite=False) -> None:
        """
        Load distance of event from CSV file. Can be used for "aligned"
        distances that are not in the database
        :param: fnam_csv: path to CSV file with distances
        :param: overwrite: Overwrite existing location from BED?
        """
        from csv import DictReader
        with open(fnam_csv, 'r') as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                if overwrite or (self.distance is None):
                    if self.name == row['name']:
                        self.distance = float(row['distance'])
                        self.origin_time = utct(row['time'])
                        self.distance_type = 'aligned'
                        # print('Found aligned distance %f for event %s' %
                        #       (self.distance, self.name))

    def calc_distance(self,
                      vp: float = CRUST_VP,
                      vs: float = CRUST_VS) -> Union[float, None]:
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
            return kilometers2degrees(distance_km, radius=RADIUS_MARS)
        else:
            return None

    def read_waveforms(self,
                       inv: obspy.Inventory,
                       sc3dir: str,
                       event_tmp_dir='./events',
                       kind: str = 'DISP',
                       fmin_SP: float = 0.5,
                       fmin_VBB: float = 1. / 30.) -> None:
        """
        Wrapper to check whether local copy of corrected waveform exists and
        read it from sc3dir otherwise (and create local copy)
        :param inv: Obspy.Inventory to use for instrument correction
        :param sc3dir: path to data, in SeisComp3 directory structure
        :param kind: 'DISP', 'VEL' or 'ACC'. Note that many other functions
                     expect the data to be in displacement
        """
        if not self.read_data_local(dir_cache=event_tmp_dir):
            self.read_data_from_sc3dir(inv, sc3dir, kind,
                                       fmin_SP=fmin_SP,
                                       fmin_VBB=fmin_VBB)
            self.write_data_local(dir_cache=event_tmp_dir)

        if self.baz is not None:
            self.add_rotated_traces()
        self._waveforms_read = True
        self.kind = 'DISP'

    def add_rotated_traces(self):
        # Add rotated phases to waveform objects
        st_rot = self.waveforms_VBB.copy()
        st_rot.rotate('NE->RT', back_azimuth=self.baz)
        for chan in ['?HT', '?HR']:
            self.waveforms_VBB += st_rot.select(channel=chan)[0]

        if self.waveforms_SP is not None:
            st_rot = self.waveforms_SP.copy()
            st_rot.rotate('NE->RT', back_azimuth=self.baz)
            for chan in ['?HT', '?HR']:
                self.waveforms_SP += st_rot.select(channel=chan)[0]

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
        VBB_path = pjoin(waveform_path, 'waveforms_VBB.mseed')
        SP_path = pjoin(waveform_path, 'waveforms_SP.mseed')
        if len(glob(origin_path)) > 0:
            with open(origin_path, 'r') as f:
                origin_local = f.readline().strip()
            if origin_local == self.origin_publicid:
                if len(glob(VBB_path)):
                    self.waveforms_VBB = obspy.read(VBB_path)
                    success = True
                else:
                    self.waveforms_VBB = None

                if len(glob(SP_path)):
                    self.waveforms_SP = obspy.read(SP_path)
                    success = True
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
        if self.waveforms_VBB is not None and len(self.waveforms_VBB) > 0:
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
                              fmin_SP=0.5,
                              fmin_VBB=1. / 30.,
                              tpre_SP: float = 100,
                              tpre_VBB: float = 1200.) -> None:
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

        filenam_SP_HG = 'XB.ELYSE.65.EH?.D.%04d.%03d'
        fnam_SP = create_fnam_event(
            filenam_inst=filenam_SP_HG,
            sc3dir=sc3dir, time=self.picks['start'])

        if len(glob(fnam_SP)) > 0:
            # Use SP waveforms only if 65.EH? exists, not otherwise (we
            # don't need 20sps SP data)
            self.waveforms_SP = read_data(fnam_SP, inv=inv, kind=kind,
                                          twin=[twin_start - tpre_SP,
                                                twin_end + tpre_SP],
                                          fmin=fmin_SP)
        else:
            self.waveforms_SP = None

        # Try for 02.BH? (20sps VBB)
        success_VBB = False
        filenam_VBB_HG = 'XB.ELYSE.02.BH?.D.%04d.%03d'
        fnam_VBB = create_fnam_event(
            filenam_inst=filenam_VBB_HG,
            sc3dir=sc3dir, time=self.picks['start'])
        if len(glob(fnam_VBB)) % 3 == 0:
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           fmin=fmin_VBB,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True

        if not success_VBB:
            # Try for 03.BH? (10sps VBB)
            filenam_VBB_HG = 'XB.ELYSE.03.BH?.D.%04d.%03d'
            fnam_VBB = create_fnam_event(
                filenam_inst=filenam_VBB_HG,
                sc3dir=sc3dir, time=self.picks['start'])
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           fmin=fmin_VBB,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True

        if not success_VBB:
            # Try for 15.BL? (10sps VBB)
            filenam_VBB_HG = 'XB.ELYSE.15.HL?.D.%04d.%03d'
            fnam_VBB = create_fnam_event(
                filenam_inst=filenam_VBB_HG,
                sc3dir=sc3dir, time=self.picks['start'])
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           fmin=fmin_VBB,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True
                
        if not success_VBB:
            # Try for 07.BL? (20sps VBB at low gain)
            filenam_VBB_HG = 'XB.ELYSE.07.BL?.D.%04d.%03d'
            fnam_VBB = create_fnam_event(
                filenam_inst=filenam_VBB_HG,
                sc3dir=sc3dir, time=self.picks['start'])
            self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                           kind=kind,
                                           fmin=fmin_VBB,
                                           twin=[twin_start - tpre_VBB,
                                                 twin_end + tpre_VBB])
            if len(self.waveforms_VBB) == 3:
                success_VBB = True

        if not success_VBB:
            self.waveforms_VBB = None

        if self.waveforms_VBB is None and self.waveforms_SP is None:
            raise FileNotFoundError('Neither SP nor VBB data found on day %s' %
                                    self.picks['start'])

    def available_sampling_rates(self):
        available = dict()
        channels = {'VBB_Z': '??Z',
                    'VBB_N': '??N',
                    'VBB_E': '??N'}
        for chan, seed in channels.items():
            if self.waveforms_VBB is None:
                available[chan] = 0.0
            else:
                available[chan] = self.waveforms_VBB.select(
                    channel=seed)[0].stats.sampling_rate

        channels = {'SP_Z': 'EHZ',
                    'SP_N': 'EHN',
                    'SP_E': 'EHE'}

        for chan, seed in channels.items():
            if self.waveforms_SP is None:
                available[chan] = None
            else:
                st = self.waveforms_SP.select(channel=seed)
                if len(st) > 0:
                    available[chan] = st[0].stats.sampling_rate
                else:
                    available[chan] = None
        return available

    def calc_spectra(self, winlen_sec, detick_nfsamp=0):
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
            if self.waveforms_VBB is not None:
                for chan in ['Z', 'N', 'E']:
                    st_sel = self.waveforms_VBB.select(
                        channel='??' + chan).copy()
                    tr = detick(st_sel[0], detick_nfsamp=detick_nfsamp)
                    tr.trim(starttime=utct(twin[0]),
                            endtime=utct(twin[1]))

                    if tr.stats.npts > 0:
                        f, p = calc_PSD(tr, winlen_sec=winlen_sec)
                        spectrum_variable['p_' + chan] = p
                        spectrum_variable['f'] = f

                if len(spectrum_variable) > 0:
                    self.spectra[variable] = spectrum_variable

            if self.waveforms_SP is not None:
                spectrum_variable = dict()
                for chan in ['Z', 'N', 'E']:
                    st_sel = self.waveforms_SP.select(
                        channel='??' + chan).copy()
                    if len(st_sel) > 0:
                        tr = detick(st_sel[0], detick_nfsamp=detick_nfsamp)
                        tr.trim(starttime=utct(twin[0]),
                                endtime=utct(twin[1]))

                        if tr.stats.npts > 0:
                            f, p = calc_PSD(tr, winlen_sec=winlen_sec)
                            spectrum_variable['p_' + chan] = p
                            spectrum_variable['f'] = f
                    else:
                        # Case that only SP1==SPZ is switched on
                        spectrum_variable['p_' + chan] = \
                            np.zeros_like(p)

            if len(spectrum_variable) > 0:
                self.spectra_SP[variable] = spectrum_variable
            if self.waveforms_VBB is None and self.waveforms_SP is not None:
                self.spectra[variable] = spectrum_variable

        # compute horizontal spectra on VBB
        for signal in self.spectra.keys():
            if signal in self.spectra:
                self.spectra[signal]['p_H'] = \
                    self.spectra[signal]['p_N'] + self.spectra[signal]['p_E']

        # compute horizontal spectra on SP
        for signal in self.spectra_SP.keys():
            if signal in self.spectra:
                self.spectra_SP[signal]['p_H'] = \
                    self.spectra_SP[signal]['p_N'] + self.spectra_SP[signal][
                        'p_E']

        self.amplitudes = {'A0': None,
                           'tstar': None,
                           'A_24': None,
                           'f_24': None,
                           'f_c': None,
                           'width_24': None}

        if 'noise' in self.spectra:
            f_noise = self.spectra['noise']['f']
            p_noise = self.spectra['noise']['p_Z']
            for signal in ['S', 'P', 'all']:
                amplitudes = None
                if signal in self.spectra:
                    if self.mars_event_type_short == 'SF':
                        comp = 'p_H'
                    else:
                        comp = 'p_Z'

                    p_sig = None
                    if comp in self.spectra[signal]:
                        p_sig = self.spectra[signal][comp]
                    elif comp in self.spectra_SP[signal]:
                        p_sig = self.spectra_SP[signal][comp]
                    if p_sig is not None:
                        f_sig = self.spectra[signal]['f']
                    amplitudes = fit_spectra(f_sig=f_sig,
                                             f_noise=f_noise,
                                             p_sig=p_sig,
                                             p_noise=p_noise,
                                             event_type=self.mars_event_type_short)
                if amplitudes is not None:
                    break
            if amplitudes is not None:
                self.amplitudes = amplitudes

        self._spectra_available = True

    def pick_amplitude(self,
                       pick: str,
                       comp: str,
                       fmin: float,
                       fmax: float,
                       instrument: str = 'VBB',
                       twin_sec: float = 10.,
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
        :param twin_sec: time window around amplitude pick in which to look
                         for maximum amplitude.
        :param unit: 'm', 'nm', 'pm', 'fm'
        :return: amplitude in time window around pick time
        """
        if not self._waveforms_read:
            raise RuntimeError('waveforms not read in Event object\n' +
                               'Call Event.read_waveforms() first.')

        if instrument == 'VBB':
            if self.waveforms_VBB is None:
                return None
            else:
                st_work = self.waveforms_VBB.copy()
        else:
            st_work = self.waveforms_SP.copy()

        st_work.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)

        if unit == 'nm':
            output_fac = 1e9
        elif unit == 'pm':
            output_fac = 1e12
        elif unit == 'fm':
            output_fac = 1e15
        elif unit == 'm':
            output_fac = 1.
        else:
            raise ValueError('Unknown unit %s' % unit)

        if not self.kind == 'DISP':
            raise RuntimeError('Waveform must be displacement for amplitudes')

        if self.picks[pick] == '':
            return None
        else:
            tmin = utct(self.picks[pick]) - twin_sec
            tmax = utct(self.picks[pick]) + twin_sec
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
                 'MFB': mag.MFB,
                 'MFB_HF': mag.MFB_HF
                 }
        if self.distance is None and distance is None:
            return None
        elif self.distance is not None and distance is None:
            distance = self.distance
        else:
            distance = distance
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
            if self.mars_event_type_short in ['24', 'HF', 'VF']:
                mag_type = 'MFB_HF'
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

    def plot_envelope(self, comp='Z',
                      figsize=(4, 3),
                      t0=0.0,
                      starttime=None, endtime=None,
                      fmin=0.05, fmax=10.,
                      ax=None):

        import matplotlib.pyplot as plt
        from mqs_reports.utils import envelope_smooth
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=figsize)
            new_ax = True
        else:
            new_ax = False

        tr = self.waveforms_VBB.select(channel='??' + comp)[0].copy()
        if starttime is not None:
            tr.trim(starttime=starttime)
        if endtime is not None:
            tr.trim(endtime=endtime)

        tr.differentiate()
        tr.differentiate()
        tr.filter('highpass', freq=fmin, corners=8)
        tr.filter('lowpass', freq=fmax, corners=8)

        tr_env = envelope_smooth(envelope_window_in_sec=10., tr=tr)

        ax.plot(tr_env.times() + t0,
                tr_env.data * 1e9)
        ax.axvline(x=0., color='k', zorder=5
                   )
        # ax.text(x=10., y=fmax * 0.9, s='P',
        #         bbox=dict(edgecolor='black',
        #                   facecolor='white',
        #                   alpha=0.5),
        #         fontsize=14)
        ax.axvline(x=utct(self.picks['S']) - utct(self.picks['P']), color='k',
                   zorder=3)
        # ax.text(x=utct(self.picks['S']) - utct(self.picks['P']) + 10.,
        #         bbox=dict(edgecolor='black',
        #                   facecolor='white',
        #                   alpha=0.5),
        #         y=fmax * 0.9, s='S', fontsize=14)
        if new_ax:
            plt.show()

    def plot_spectrogram(self, comp='Z',
                         figsize=(4, 3),
                         kind='cwt',
                         t0=0.0,
                         starttime=None, endtime=None,
                         fmin=0.05, fmax=10.,
                         ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=figsize)
            new_ax = True
        else:
            new_ax = False

        tr = self.waveforms_VBB.select(channel='??' + comp)[0].copy()
        if starttime is not None:
            tr.trim(starttime=starttime)
        if endtime is not None:
            tr.trim(endtime=endtime)

        tr = detick(tr=tr, detick_nfsamp=5)

        tr.differentiate()
        tr.differentiate()
        z, f, t = calc_cwf(tr,
                           fmin=fmin, fmax=fmax)
        # z, f, t = calc_specgram(tr, fmin=fmin, fmax=fmax)

        z = 10 * np.log10(z)
        z[z < -210] = -210.
        z[z > -160] = -160.
        # df = 2
        # dt = 4
        # ax.pcolormesh(t[::dt], f[::df],z[::df, ::dt], vmin=-220, vmax=-150)
        ax.pcolormesh(t + t0, f, z, vmin=-210, vmax=-160,
                      rasterized=True)
        ax.axvline(x=0., color='k', zorder=5
                   )
        ax.text(x=10., y=fmax * 0.95, s='P',
                verticalalignment='top',
                # bbox=dict(edgecolor='black',
                #          facecolor='white',
                #          alpha=0.5),
                fontsize=14)
        ax.axvline(x=utct(self.picks['S']) - utct(self.picks['P']), color='k',
                   zorder=3)
        ax.text(x=utct(self.picks['S']) - utct(self.picks['P']) + 10.,
                verticalalignment='top',
                # bbox=dict(edgecolor='black',
                #          facecolor='white',
                #          alpha=0.5),
                y=fmax * 0.95, s='S', fontsize=14)
        if new_ax:
            plt.show()

    def plot_spectrum(self, comp='Z',
                      window: str = 'S',
                      figsize=(4, 3),
                      color_spec='red',
                      color_noise='black',
                      plot_fit=False,
                      flip_axes=False,
                      ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=figsize)
            new_ax = True
        else:
            new_ax = False

        x = self.spectra[window]['f']
        y = 10. * np.log10(self.spectra[window]['p_' + comp])
        if flip_axes:
            ax.plot(y, x, c=color_spec)
        else:
            ax.plot(x, y, c=color_spec)

        y = 10. * np.log10(self.spectra['noise']['p_' + comp])
        if flip_axes:
            ax.plot(y, x, c=color_noise)
        else:
            ax.plot(x, y, c=color_noise)

        if flip_axes:
            ax.set_ylim(0., 2.)
            ax.set_xlim(-230., -160.)
            ax.set_ylabel('frequency / Hz')
            ax.set_xlabel('power spectral density / m$^2$/Hz')
        else:
            ax.set_xlim(0., 2.)
            ax.set_ylim(-230., -160.)
            ax.set_xlabel('frequency / Hz')
            ax.set_ylabel('power spectral density / m$^2$/Hz')
        ax.set_title('Spectrum %s' % self.name)

        if plot_fit:
            f = np.geomspace(0.01, 10., 100)
            f_c = 1.0
            stf_amp = 1. / (1. + (f / f_c) ** 2) ** 2
            y = self.amplitudes['A0'] + 10 * np.log10(
                    np.exp(- np.pi * self.amplitudes['tstar'] * f)
                    * stf_amp)
            if flip_axes:
                ax.plot(y, f)
            else:
                ax.plot(f, y)

        if new_ax:
            plt.tight_layout()
            plt.show()

    def plot_waveform(self, comp='Z',
                      window: str = 'S',
                      figsize=(4, 3),
                      color_spec='red',
                      color_noise='black',
                      fmin=None, fmax=None,
                      ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=figsize)
            new_ax = True
        else:
            new_ax = False

        tr_work = self.waveforms_VBB.select(channel='??' + comp)[0]
        tr_work.differentiate()
        tr_work.decimate(2)
        tr_work.trim(starttime=utct(self.picks['P']) - 60.,
                     endtime=utct(self.picks['P']) + 520.)

        ax.plot(tr_work.times() - 60., tr_work.data,
                lw=0.5)

        offset = np.quantile(abs(tr_work.data), q=0.99)

        ax.axvline(x=0., color='k', zorder=-1
                   )
        ax.text(x=10., y=-offset * 1.1, s='P',
                bbox=dict(edgecolor='black',
                          facecolor='white',
                          alpha=0.5),
                fontsize=14)
        ax.axvline(x=utct(self.picks['S']) - utct(self.picks['P']), color='k',
                   zorder=-1)
        ax.text(x=utct(self.picks['S']) - utct(self.picks['P']) + 10.,
                bbox=dict(edgecolor='black',
                          facecolor='white',
                          alpha=0.5),
                y=-offset * 1.1, s='S', fontsize=14)
        ax.text(0.12, 0.95,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                s='filtered, %3.1f-%3.1f Hz' % (fmin, fmax))
        ax.text(0.12, 0.05,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                s='raw')

        if fmin is not None and fmax is not None:
            tr_work.filter('highpass', freq=fmin)
            tr_work.filter('lowpass', freq=fmax)
            ax.plot(tr_work.times() - 60., tr_work.data + offset * 1.5,
                    lw=0.5)

        if new_ax:
            plt.tight_layout()
            plt.show()

    def make_report(self, chan, fnam_out, annotations=None):
        from mqs_reports.report import make_report
        make_report(self, chan=chan, fnam_out=fnam_out, annotations=annotations)

    def write_locator_yaml(self, fnam_out, dt=2.):
        with open(fnam_out, 'w') as f:
            f.write('velocity_model: MQS_Ops.2019-01-03_250\n')
            f.write('velocity_model_uncertainty: 1.5\n')
            if self.distance_type == 'GUI':
                f.write('backazimuth:\n')
                f.write(f'    value: {self.baz}\n')
            f.write('phases:\n')
            for pick, pick_time in self.picks.items():
                if pick in ('P', 'S', 'PP', 'SS', 'pP', 'sS', 'ScS'):
                    f.write(' -\n')
                    f.write(f'    code: {pick}\n')
                    f.write(f'    datetime: {pick_time}\n')
                    f.write(f'    uncertainty_lower: {dt}\n')
                    f.write(f'    uncertainty_upper: {dt}\n')
                    f.write(f'    uncertainty_model: uniform\n')
                    f.write('\n')

    def rotation_plot(self, angles, fmin, fmax):
        import matplotlib.pyplot as plt
        from mqs_reports.utils import envelope_smooth
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all',
                               figsize=(10, 6))

        nangles = len(angles)
        st_work = self.waveforms_VBB.select(channel='??[ENZ]').copy()
        st_work.decimate(5)
        st_work.filter('highpass', freq=fmin, corners=6)
        st_work.filter('lowpass', freq=fmax, corners=6)
        st_work.trim(starttime=utct(self.origin_time) - 50.,
                     endtime=utct(self.origin_time) + 850.)

        for iangle, angle in enumerate(angles):

            st_rot: obspy.Stream = st_work.copy()
            st_rot.rotate('NE->RT', back_azimuth=angle)

            tr_R_env = envelope_smooth(tr=st_rot.select(channel='BHR')[0],
                                       envelope_window_in_sec=10.)
            tr_T_env = envelope_smooth(tr=st_rot.select(channel='BHT')[0],
                                       envelope_window_in_sec=10.)
            tr_Z_env = envelope_smooth(tr=st_rot.select(channel='BHZ')[0],
                                       envelope_window_in_sec=10.)
            maxfac = np.quantile(tr_Z_env.data, q=0.98)
            for itr, tr in enumerate((tr_R_env, tr_T_env)):
                xvec = tr_Z_env.times() + float(tr_Z_env.stats.starttime - \
                                                utct(self.picks['P']))
                ax[itr].plot(xvec,
                             iangle + tr_Z_env.data / maxfac, c='grey',
                             lw=1)
                ax[itr].fill_between(x=xvec,
                                     y1=iangle + tr_Z_env.data / maxfac,
                                     y2=iangle, color='darkgrey')
                ax[itr].plot(xvec,
                             iangle + tr.data / maxfac, c='k', lw=1.5,
                             zorder=50)

        self.mark_phases(ax, tref=utct(self.picks['P']))
        ax[0].set_yticks(range(0, nangles))
        ax[0].set_yticklabels(angles)
        ax[0].set_xlim(-50, 550)
        ax[0].set_ylim(-1, nangles * 1.15)
        ax[0].set_xlabel('time after P-wave')
        ax[0].set_ylabel('Rotation angle')
        ax[0].set_title('Radial component')
        ax[1].set_title('Transversal component')
        fig.suptitle('Event %s (%5.3f-%5.3f Hz)' %
                     (self.name, fmin, fmax))
        fig.savefig('rotations_%s_%3.1f_%3.1f_sec.png' %
                    (self.name, 1. / fmax, 1. / fmin),
                    dpi=200)

    def mark_phases(self, ax, tref):
        for a in ax:
            for pick in ['P', 'S', 'Pg', 'Sg']:
                try:
                    a.axvline(utct(self.picks[pick]) - tref,
                              c='darkred', ls='dashed')
                except TypeError:
                    pass
            for pick in ['start', 'end']:
                a.axvline(utct(self.picks[pick]) - tref,
                          c='darkgreen', ls='dashed')

    def plot_filterbank(self,
                        fmin: float = 1. / 64,
                        fmax: float = 4.,
                        df: float = 2 ** 0.5,
                        log: bool = False,
                        waveforms: bool = False,
                        normwindow: str = 'all',
                        annotations: Annotations = None,
                        tmin_plot: float = None,
                        tmax_plot: float = None,
                        timemarkers: dict = None,
                        starttime: obspy.UTCDateTime = None,
                        endtime: obspy.UTCDateTime = None,
                        instrument: str = 'VBB',
                        fnam: str = None):
        import matplotlib.pyplot as plt
        import warnings
        from mqs_reports.utils import envelope_smooth

        def mark_glitch(ax: list,
                        x0: float, x1: float,
                        ymin: float = -2.,
                        height: float = 50., **kwargs):
            from matplotlib.patches import Rectangle
            xy = [x0, ymin]
            width = x1 - x0
            for a in ax:
                rect = Rectangle(xy=xy, width=width, height=height, **kwargs)
                a.add_patch(rect)

        fig, ax = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all',
                               figsize=(10, 6))

        # Determine frequencies
        nfreqs = int(np.round(np.log(fmax / fmin) / np.log(df), decimals=0) + 1)
        freqs = np.geomspace(fmin, fmax + 0.001, nfreqs)

        # Reference time
        if 'P' in self.picks and len(self.picks['P']) > 0:
            t_ref = utct(self.picks['P'])
            t_ref_type = 'P'
        else:
            t_ref = self.starttime
            t_ref_type = 'start time'

        if instrument == 'VBB':
            st_work = self.waveforms_VBB.select(channel='??[ENZ]').copy()
        elif instrument == 'SP':
            try:
                st_work = self.waveforms_SP.select(channel='??[ENZ]').copy()
            except AttributeError:
                st_work = self.waveforms_VBB.select(channel='??[ENZ]').copy()
        else:
            raise ValueError(f'Invalid value for instrument: {instrument}')

        try:
            st_work.rotate('NE->RT', back_azimuth=self.baz)
        except:
            rotated = False
        else:
            rotated = True

        tstart_norm = dict(P=self.picks['P_spectral_start'],
                           S=self.picks['S_spectral_start'],
                           all=self.starttime)
        tend_norm = dict(P=self.picks['P_spectral_end'],
                         S=self.picks['S_spectral_end'],
                         all=self.endtime)
        if normwindow == 'S' and len(tstart_norm[normwindow]) == 0:
            normwindow = 'P'
            if len(tstart_norm[normwindow]) == 0:
                normwindow = 'all'
        tstart_norm = utct(tstart_norm[normwindow])
        tend_norm = utct(tend_norm[normwindow])

        if starttime is None:
            starttime = self.starttime - 100.
        if endtime is None:
            endtime = self.endtime + 100.
        if tmin_plot is None:
            tmin_plot = starttime - t_ref
        if tmax_plot is None:
            tmax_plot = endtime - t_ref

        st_work.trim(starttime=utct(starttime) - 1. / fmin,
                     endtime=utct(endtime) + 1. / fmin)

        for ifreq, fcenter in enumerate(freqs):
            f0 = fcenter / df
            f1 = fcenter * df
            st_filt = st_work.copy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    st_filt.filter('bandpass', freqmin=f0, freqmax=f1,
                                   corners=8)
            except ValueError:  # If f0 is above Nyquist
                print('No 20sps data available for event %s' % self.name)
            else:
                st_filt.trim(starttime=utct(starttime),
                             endtime=utct(endtime))

                if rotated:
                    tr_3 = st_filt.select(channel='?HT')[0]
                    tr_2 = st_filt.select(channel='?HR')[0]
                else:
                    tr_2 = st_filt.select(channel='?HN')[0]
                    tr_3 = st_filt.select(channel='?HE')[0]
                tr_2_env = envelope_smooth(tr=tr_2, mode='same',
                                           envelope_window_in_sec=10.)
                tr_3_env = envelope_smooth(tr=tr_3, mode='same',
                                           envelope_window_in_sec=10.)
                tr_Z = st_filt.select(channel='?HZ')[0]
                tr_Z_env = envelope_smooth(tr=tr_Z, mode='same',
                                           envelope_window_in_sec=10.)

                tr_real = [tr_Z, tr_2, tr_3]
                for itr, tr in enumerate((tr_Z_env, tr_2_env, tr_3_env)):
                    if log:
                        tr_norm = tr.slice(starttime=tstart_norm,
                                           endtime=tend_norm)
                        maxfac = np.quantile(tr_norm.data, q=0.9)
                        offset = np.quantile(tr_norm.data, q=0.1)
                    else:
                        tr_norm = tr.slice(starttime=tstart_norm,
                                           endtime=tend_norm,
                                           nearest_sample=True)
                        try: 
                            maxfac = np.quantile(tr_norm.data, q=0.9)
                            offset = np.quantile(tr_norm.data, q=0.1)
                        except:
                            maxfac = 1.e-9
                            offset = 0.

                    t_offset = float(tr_Z_env.stats.starttime - t_ref)
                    xvec_env = tr_Z_env.times() + t_offset
                    xvec = tr_Z.times() + t_offset
                    # ax[itr].plot(xvec_env,
                    #              iangle + tr_Z_env.data / maxfac, c='grey',
                    #              lw=1)
                    # ax[itr].fill_between(x=xvec_env,
                    #                      y1=iangle + tr_Z_env.data / maxfac,
                    #                      y2=iangle, color='darkgrey')
                    if log:
                        ax[itr].plot(xvec_env,
                                     ifreq + np.log(tr.data / maxfac) / 3,
                                     lw=1.0, zorder=50)
                    else:
                        if waveforms:
                            color = 'k'
                        else:
                            color = 'C%d' % (ifreq % 10)

                        ax[itr].plot(xvec_env,
                                     ifreq + (tr.data - offset) / maxfac,
                                     c=color,
                                     lw=0.5, zorder=80)
                        if waveforms:
                            ax[itr].plot(xvec,
                                         ifreq + tr_real[itr].data / maxfac,
                                         c='C%d' % (ifreq % 10),
                                         lw=0.5, zorder=50 - ifreq)

        if timemarkers is not None:
            for phase, time in timemarkers.items():
                if tmin_plot < time < tmax_plot:
                    for a in ax:
                        a.axvline(x=time, ls='dashed')
                        a.text(x=time, y=nfreqs, s=phase)

        self.mark_phases(ax, tref=t_ref)

        if annotations is not None:
            annotations_event = annotations.select(
                starttime=utct(self.picks['start']) - 180.,
                endtime=utct(self.picks['end']) + 180.)
            if len(annotations_event) > 0:
                x0s = []
                x1s = []
                for times in annotations_event:
                    tmin_glitch = utct(times[0])
                    tmax_glitch = utct(times[1])
                    x0s.append(
                        float(tmin_glitch) - float(t_ref))
                    x1s.append(
                        float(tmax_glitch) - float(t_ref))

                for x0, x1 in zip(x0s, x1s):
                    mark_glitch(ax, x0, x1, fc='lightgrey',
                                zorder=-3, alpha=0.8)
            mark_glitch(ax,
                        x0=tstart_norm - float(t_ref),
                        x1=tend_norm - float(t_ref),
                        ymin=-1, height=0.3, fc='grey', alpha=0.8
                        )
        ax[0].set_yticks(range(0, nfreqs))
        np.set_printoptions(precision=3)
        ticklabels = []
        for freq in freqs:
            if freq > 1:
                ticklabels.append(f'{freq:.1f}Hz')
            else:
                ticklabels.append(f'1/{1. / freq:.1f}Hz')
        ax[0].set_yticklabels(ticklabels)
        for a in ax:
            # a.set_xticks(np.arange(-300, 1000, 100), minor=False)
            a.set_xticks(np.arange(-300, 3000, 25), minor=True)
            if t_ref_type == 'P':
                a.set_xlabel('time after P-wave')
            else:
                a.set_xlabel('time after start time')
            a.grid(b=True, which='both', axis='x', lw=0.2, alpha=0.3)
            a.grid(b=True, which='major', axis='y', lw=0.2, alpha=0.3)
            a.axhline(y=np.argmin(abs(freqs - 1.)),
                      ls='dashed', lw=1.0, c='k')
        ax[0].set_xlim(tmin_plot, tmax_plot)
        ax[0].set_ylim(-1.5, nfreqs + 1.5)
        ax[0].set_ylabel('frequency')
        ax[0].set_title('Vertical')
        if rotated:
            ax[1].set_title('Radial')
            ax[2].set_title('Transverse')
        else:
            ax[1].set_title('North/South')
            ax[2].set_title('East/West')
        fig.suptitle('Event %s (%5.3f-%5.3f Hz)' %
                     (self.name, fmin, fmax))
        plt.subplots_adjust(top=0.911,
                            bottom=0.097,
                            left=0.089,
                            right=0.972,
                            hspace=0.2,
                            wspace=0.116)
        if fnam is None:
            plt.show()
        else:
            fig.savefig(fnam,
                        dpi=200)
        plt.close()

    def plot_filterbank_phase(self,
                              comp: str,
                              starttime: obspy.UTCDateTime,
                              endtime: obspy.UTCDateTime,
                              tmin_plot: obspy.UTCDateTime,
                              tmax_plot: obspy.UTCDateTime,
                              tmin_amp: obspy.UTCDateTime,
                              tmax_amp: obspy.UTCDateTime,
                              ax_fbs,
                              zerophase=False,
                              df: float = 2 ** 0.5,
                              waveforms: bool = False,
                              fmin=1. / 16., fmax=2.):
        import warnings
        from mqs_reports.utils import envelope_smooth
        import scipy.signal as signal

        # Determine frequencies
        nfreqs = int(np.round(np.log(fmax / fmin) /
                              np.log(df),
                              decimals=0) + 1)
        freqs = np.geomspace(fmin, fmax + 0.001, nfreqs)

        # Reference time
        if 'P' in self.picks and len(self.picks['P']) > 0:
            t_ref = utct(self.picks['P'])
            t_ref_type = 'P'
        else:
            t_ref = self.starttime
            t_ref_type = 'start time'

        if len(self.waveforms_VBB.select(channel='?HT')) == 0:
            self.add_rotated_traces()
        st_work = self.waveforms_VBB.select(channel='??[RTENZ]').copy()

        tstart_norm = utct(starttime)
        tend_norm = utct(endtime)

        if tmin_plot is None:
            tmin_plot = starttime - t_ref
            tmax_plot = endtime - t_ref

        st_work.trim(starttime=utct(starttime) - 1. / fmin,
                     endtime=utct(endtime) + 1. / fmin)

        envs_out = np.zeros(nfreqs)

        for ifreq, fcenter in enumerate(freqs):
            f0 = fcenter / df
            f1 = fcenter * df
            st_filt = st_work.copy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if zerophase:
                        corners = 3
                    else:
                        corners = 6
                    f0_norm = f0 / (st_filt[0].stats.sampling_rate / 2.)
                    f1_norm = f1 / (st_filt[0].stats.sampling_rate / 2.)

                    bh, ah = signal.butter(N=corners,
                                           Wn=(f0_norm),
                                           btype='highpass')

                    w, h = signal.freqz(b=bh, a=ah, worN=2 ** 14)
                    bl, al = signal.butter(N=corners,
                                           Wn=(f1_norm),
                                           btype='lowpass')
                    w2, h2 = signal.freqz(b=bl, a=al, worN=2 ** 14)
                    for tr in st_filt:
                        if zerophase:
                            tr.data = signal.filtfilt(bh, ah, tr.data)
                            tr.data = signal.filtfilt(bl, al, tr.data)
                            resp = np.trapz(y=(abs(h) * abs(h2)) ** 2.,
                                            x=w / (2 * np.pi) *
                                              tr.stats.sampling_rate)
                        else:
                            signal.lfilter(bh, ah, tr.data)
                            signal.lfilter(bl, al, tr.data)
                            resp = np.trapz(y=abs(h) * abs(h2),
                                            x=w / (2 * np.pi) *
                                              tr.stats.sampling_rate)

                    # st_filt.filter('bandpass',
                    #                freqmin=f0, freqmax=f1,
                    #                zerophase=zerophase,
                    #                corners=corners)
            except ValueError:  # If f0 is above Nyquist
                print('No 20sps data available for event %s' % self.name)
            else:
                st_filt.trim(starttime=utct(starttime),
                             endtime=utct(endtime))
                tr = st_filt.select(channel='?H' + comp)[0]
                tr_env = envelope_smooth(tr=tr, mode='same',
                                         envelope_window_in_sec=5.)

                tr_norm = tr.slice(starttime=tstart_norm,
                                   endtime=tend_norm,
                                   nearest_sample=True)
                # try:
                #    maxfac = np.quantile(tr_norm.data, q=0.9)
                #    offset = np.quantile(tr_norm.data, q=0.1)
                # except:
                maxfac = 6.e-11
                maxfac = np.quantile(tr_env.data, q=0.5)
                offset = np.quantile(tr_env.data, q=0.1)
                # offset = 0.

                t_offset = float(tr_env.stats.starttime - t_ref)
                xvec_env = tr_env.times() + t_offset
                xvec = tr.times() + t_offset
                if waveforms:
                    color = 'k'
                else:
                    color = 'C%d' % (ifreq % 10)

                ax_fbs.plot(xvec_env,
                            ifreq + (tr_env.data - offset) / maxfac,
                            c=color,
                            lw=0.5, zorder=80)

                tr_env_amp = tr_env.slice(starttime=t_ref + tmin_amp,
                                          endtime=t_ref + tmax_amp)

                xvec_env_amp = tr_env_amp.times() + tmin_amp
                envs_out[ifreq] = tr_env_amp.data.max() / np.sqrt(resp)
                # np.sqrt(f1 - f0)
                ax_fbs.plot(xvec_env_amp,
                            ifreq + (tr_env_amp.data - offset) / maxfac,
                            c=color,
                            lw=2.0, zorder=80)
                if waveforms:
                    ax_fbs.plot(xvec,
                                ifreq + tr.data / maxfac,
                                c='C%d' % (ifreq % 10),
                                lw=0.5, zorder=50 - ifreq)
        ax_fbs.set_yticks(range(0, nfreqs))
        np.set_printoptions(precision=3)
        ticklabels = []
        for freq in freqs:
            if freq > 1:
                ticklabels.append(f'{freq:.1f}Hz')
            else:
                ticklabels.append(f'1/{1. / freq:.1f}Hz')
        ax_fbs.set_yticklabels(ticklabels)
        ax_fbs.set_xticks(np.arange(-300, 3000, 25), minor=True)
        if t_ref_type == 'P':
            ax_fbs.set_xlabel('time after P-wave')
        else:
            ax_fbs.set_xlabel('time after start time')
        ax_fbs.grid(b=True, which='both', axis='x', lw=0.2, alpha=0.3)
        ax_fbs.grid(b=True, which='major', axis='y', lw=0.2, alpha=0.3)
        ax_fbs.axhline(y=np.argmin(abs(freqs - 1.)),
                       ls='dashed', lw=1.0, c='k')
        ax_fbs.set_xlim(tmin_plot, tmax_plot)
        ax_fbs.set_ylim(-1.5, nfreqs + 1.5)
        ax_fbs.set_ylabel('frequency')

        return freqs, envs_out

    def plot_polarization_event_noise(self, 
                                  t_pick_P, t_pick_S,
                                  rotation = 'ZNE', BAZ=0.0,
                                  kind='spec', fmin=1., fmax=10.,
                                  winlen_sec=20., overlap=0.5,
                                  tstart=None, tend=None, vmin=None,
                                  vmax=None, log=False, fname=None,
                                  dop_winlen=60, dop_specwidth=0.2,
                                  nf=100, w0=20,
                                  use_alpha=True, use_alpha2=False, 
                                  alpha_inc = 1.0, alpha_elli = 1.0, alpha_azi = 1.0,
                                  plot_6C=True, plot_spec_azi_only = False,
                                  differentiate=False, detick_1Hz=False,
                                  impact = False):
        
        """
        Plots polarisation of seismic event with window of noise and manually defined event time window
        """
        import matplotlib
        import matplotlib.patches as patches
        import matplotlib.dates as mdates
        import polarization.polarization as polarization
        from obspy import Stream
        from obspy.signal.util import next_pow_2
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import make_axes
        from matplotlib.ticker import NullFormatter
        import seaborn as sns
        
        mod_180 = False #set to True if only mapping 0-180° azimuth, False maps 0-360°
        trim_time = [60., 300.] #[time before noise start, time after S] [seconds] Trims waveform
        f_band_density = [0.3, 1.] #frequency band for density plot
        
        st_Copy = self.waveforms_VBB.copy() 
        phase_P = 'P' if self.picks['P'] else 'Pg'
        phase_S = 'S' if self.picks['S'] else 'Sg'
        
        #Rotate the waveforms into different coordinate system: ZRT or LQT
        if 'ZNE' not in rotation:
            if 'RT' in rotation:
                st_Copy.rotate('NE->RT', back_azimuth=BAZ)
                components = ['Z', 'R', 'T']
            elif 'LQT' in rotation:
                st_Copy.rotate('ZNE->LQT', back_azimuth=BAZ, inclination = 40.0)
                components = ['L', 'Q', 'T']
            else:
                raise Exception("Sorry, please pick valid rotation system: ZNE, RT, LQT") 
        else:
            components = ['Z', 'N', 'E']

    
        #differentiate waveforms
        if differentiate:
            st_Copy.differentiate()
        
        #trim the waveforms in length
        try:
            st_Copy.trim(starttime=utct(self.picks['noise_start']) - trim_time[0], #og: -50, +850
                         endtime=utct(self.picks[phase_S]) + trim_time[1])
        except ValueError: #if noise window is picked after the event
            st_Copy.trim(starttime=utct(self.picks[phase_P]) - trim_time[0], #og: -50, +850
                         endtime=utct(self.picks['noise_end']) + trim_time[0])
        
                                                               
        st_Z = Stream(traces=[st_Copy.select(component=components[0])[0]])
        st_N = Stream(traces=[st_Copy.select(component=components[1])[0]])
        st_E = Stream(traces=[st_Copy.select(component=components[2])[0]])
        
        tstart_signal_P = utct(self.picks[phase_P]) + t_pick_P[0]
        tend_signal_P = utct(self.picks[phase_S]) - 20 if (utct(self.picks[phase_P]) + t_pick_P[1]) > (utct(self.picks[phase_S]) - 1) else  utct(self.picks[phase_P]) + t_pick_P[1]
        # tend_signal_P = utct(self.picks[phase_P]) + t_pick_P[1]
        
        tstart_signal_S = utct(self.picks[phase_S]) + t_pick_S[0]
        tend_signal_S = utct(self.picks[phase_S]) + t_pick_S[1]
        
        # #manual adjustment list
        # if self.name == 'S0734a' or self.name == 'S0756a':
        #     tend_signal_S = utct(self.picks[phase_S]) + 50
        
        #Noise window: MQS picks
        tstart_noise = utct(self.picks['noise_start']) # -120
        tend_noise = utct(self.picks['noise_end'])
    
        tstart, tend, dt = polarization._check_traces(st_Z, st_N, st_E, tstart, tend)
    
        #define in which row Signal and Noise hist are plotted
        signal_P_row = 2
        signal_S_row = 3
        noise_row = 1
        density_row = 4
        
        # Create figure to plot in
        if plot_6C: #not tested
            gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],   # specgram, hist2d
                               height_ratios=[1, 1, 1, 1, 1, 1],
                               top=0.95,
                               bottom=0.05,
                               left=0.02,
                               right=0.91,
                               hspace=0.25,
                               wspace=0.05)
            nrows = 6
            dy_lmst = -0.4
            figsize_y = 9
        elif plot_spec_azi_only:
            gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],   # specgram, hist2d
                               height_ratios=[1, 1],
                               top=0.93,
                               bottom=0.1,
                               left=0.05,
                               right=0.89,
                               hspace=0.3,
                               wspace=0.08)
            nrows = 2
            dy_lmst = -0.4
            figsize_y = 5
        else:
            gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],  # specgram, hist2d, hist2d
                               height_ratios=[1, 1, 1, 1],
                               top=0.96,
                               bottom=0.05,
                               left=0.04, #0.05
                               right=0.94, #0.89
                               hspace=0.15,
                               wspace=0.1) #0.02 original
            nrows = 4
            dy_lmst = -0.25
            figsize_y = 9
        # dx_cbar = 0.055
        dx_cbar = 0.02
        w_cbar = 0.005
        
        gridspec_kw['top'] = 0.91
        title = f'{self.name}'
        
        fig, axes = plt.subplots(nrows=nrows, ncols=5, #sharey='all',
                                 figsize=(19, figsize_y), gridspec_kw=gridspec_kw) #16 og - 12 for 3 panel/row
        
        #join y-axis (frequency) for all but right most column
        for ax in axes[0:-1, 0].flatten():
            ax.get_shared_y_axes().join(ax, axes[-1, 0])
        for ax in axes[:, 1:-1].flatten():
            ax.get_shared_y_axes().join(ax, axes[-1, 0])
        
        if impact:
            title += f' - {rotation} impact: {impact}'
        
        rect = [[None for i in range(3)] for j in range(nrows)] #prepare rectangles to mark the time windows
        color_windows = ['C0', 'C2', 'C9'] #signal P, S, noise
        for j in range(nrows):
            rect[j][0] = patches.Rectangle((utct(tstart_signal_P).datetime,fmin+0.03*fmin), 
                                           utct(tend_signal_P).datetime-utct(tstart_signal_P).datetime, 
                                           fmax-fmin-0.03*fmax, linewidth=2, 
                                           edgecolor=color_windows[0], fill = False) #signal
            rect[j][1] = patches.Rectangle((utct(tstart_signal_S).datetime,fmin+0.03*fmin), 
                                           utct(tend_signal_S).datetime-utct(tstart_signal_S).datetime, 
                                           fmax-fmin-0.03*fmax, linewidth=2, 
                                           edgecolor=color_windows[1], fill = False) #signal
            rect[j][2] = patches.Rectangle((utct(tstart_noise).datetime,fmin+0.03*fmin), 
                                           utct(tend_noise).datetime-utct(tstart_noise).datetime, 
                                           fmax-fmin-0.03*fmax, linewidth=2, 
                                           edgecolor=color_windows[-1], fill = False) #noise
            
            # #remove the density column from the shared y axis list
            # axis_density = axes[j,density_row]
            # axis_density.get_shared_y_axes().remove(axis_density)
            # # Create and assign new ticker
            # yticker = matplotlib.axis.Ticker()
            # axis_density.yaxis.major = yticker
            
            # # The new ticker needs new locator and formatters
            # yloc = matplotlib.ticker.AutoLocator()
            # yfmt = matplotlib.ticker.ScalarFormatter()
            
            # axis_density.autoscale()
            
            # axis_density.yaxis.set_major_locator(yloc)
            # axis_density.yaxis.set_major_formatter(yfmt)
            # axis_density.yaxis.set_minor_locator(yloc)
            # axis_density.yaxis.set_minor_formatter(yfmt)
    

        winlen = int(winlen_sec / dt)
        nfft = next_pow_2(winlen) * 2
    
        # variables for statistics
        nbins = 30 #original: 90
        nts = 0
    
        # Calculate width of smoothing windows for degree of polarization analysis
        nfsum, ntsum, dsfacf, dsfact = polarization._calc_dop_windows(
            dop_specwidth, dop_winlen, dt, fmax, fmin,
            kind, nf, nfft, overlap, winlen_sec)
    
        if kind == 'spec':
            binned_data_signal_P = np.zeros((nrows, nfft // (2 * dsfacf) + 1, nbins))
            binned_data_signal_S = np.zeros_like(binned_data_signal_P)
            binned_data_noise = np.zeros_like(binned_data_signal_P)
            
            #For histogram curve
            histogram_data_signal_P = np.zeros_like(binned_data_signal_P)
            histogram_data_signal_S = np.zeros_like(binned_data_signal_P)
        else:
            binned_data_signal_P = np.zeros((nrows, nf // dsfacf, nbins))
            binned_data_signal_S = np.zeros_like(binned_data_signal_P)
            binned_data_noise = np.zeros_like(binned_data_signal_P)
            
            #For histogram curve
            histogram_data_signal_P = np.zeros_like(binned_data_signal_P)
            histogram_data_signal_S = np.zeros_like(binned_data_signal_P)
    
        for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
            if tr_Z.stats.npts < winlen * 4:
                continue
            
            if detick_1Hz:
                tr_Z_detick = detick(tr_Z, 3)
                tr_N_detick = detick(tr_N, 3)
                tr_E_detick = detick(tr_E, 3)
                f, t, u1, u2, u3 = polarization._compute_spec(tr_Z_detick, tr_N_detick, tr_E_detick, kind, fmin, fmax,
                                             winlen, nfft, overlap, nf=nf, w0=w0)
            else:
                f, t, u1, u2, u3 = polarization._compute_spec(tr_Z, tr_N, tr_E, kind, fmin, fmax,
                                                 winlen, nfft, overlap, nf=nf, w0=w0)
    
            azi1, azi2, elli, inc1, inc2, r1, r2, P = polarization.compute_polarization(
                u1, u2, u3, ntsum=ntsum, nfsum=nfsum, dsfacf=dsfacf, dsfact=dsfact)
    
            f = f[::dsfacf]
            t = t[::dsfact]
            t += float(tr_Z.stats.starttime)
            nts += len(t)
            bol_signal_P_mask= np.array((t > tstart_signal_P, t< tend_signal_P)).all(axis=0)
            bol_signal_S_mask= np.array((t > tstart_signal_S, t< tend_signal_S)).all(axis=0)
            bol_noise_mask= np.array((t > tstart_noise, t< tend_noise)).all(axis=0)
            
            #get indexes where f lies in the defined f-band for density subplot
            idx_density = np.where((f > f_band_density[0]) & (f < f_band_density[1]))
    
    
            #Scalogram and alpha/masking of signals
            # scalogram = 10 * np.log10((r1 ** 2).sum(axis=-1))
            # alpha, alpha2 = polarization._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2)
            if alpha_inc > 0.: 
                func_inc= np.cos
                func_azi= np.sin
            else: 
                alpha_inc= -alpha_inc 
                func_inc= np.sin
                func_azi= np.cos
            
            r1_sum = (r1** 2).sum(axis=-1)
            if alpha_inc is not None: 
                r1_sum *= func_inc(inc1)**(2*alpha_inc)
            elif alpha_azi is not None:
                r1_sum *= abs(func_azi(azi1))**(2*alpha_azi)
            elif alpha_elli is not None:
                r1_sum *= (1. - elli)**(2*alpha_elli)
                
            scalogram= 10 * np.log10(r1_sum) 
            alpha, alpha2= polarization._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2) 
            if mod_180: 
                azi1= azi1% np.pi
                azi2= azi2% np.pi
            
            
            if alpha_inc is not None: 
                alpha*= func_inc(inc1)**alpha_inc
            if alpha_azi is not None:
                alpha*= abs(func_azi(azi1))**alpha_azi 
            if alpha_elli is not None: 
                alpha*= (1. - elli)**alpha_elli
            
            #Prepare x axis array (datetime)
            t_datetime = np.zeros_like(t,dtype=object)
            for i, time in enumerate(t):
                 t_datetime[i] = utct(time).datetime
    
            # plot scalogram, ellipticity, major axis azimuth and inclination
    
            iterables = [
                (scalogram, vmin, vmax, np.ones_like(alpha),
                 'amplitude\n/ dB', np.arange(vmin, vmax+1, 20), 'plasma'),
                (np.rad2deg(azi1), 0, 360, alpha,
                 'major azimuth\n/ degree', np.arange(0, 361, 90), 'tab20b'), #was 45 deg steps
                (elli, 0, 1, alpha,
                 'ellipticity\n', np.arange(0, 1.1, 0.2), 'gnuplot'),
                (np.rad2deg(abs(inc1)), -0, 90, alpha,
                 'major inclination\n/ degree', np.arange(0, 91, 20),
                 'gnuplot')]
            
            if plot_spec_azi_only:
                del iterables[-2:]
            if plot_6C:
                iterables.append(
                    (np.rad2deg(azi2), 0, 180, alpha2,
                     'minor azimuth\n/ degree', np.arange(0, 181, 30), 'hsv'))
                iterables.append(
                    (np.rad2deg(inc2), -90, 90, alpha2,
                     'minor inclination\n/ degree', np.arange(-90, 91, 30), 'hsv'))
    
            for irow, [data, rmin, rmax, a, xlabel, xticks, cmap] in \
                    enumerate(iterables):
    
                ax = axes[irow, 0]
    
                if log and kind == 'cwt':
                    # imshow can't do the log sampling in frequency
                    cm = polarization.pcolormesh_alpha(ax, t_datetime, f, data, alpha=a, cmap=cmap,
                                          vmin=rmin, vmax=rmax)
                else:
                    cm = polarization.imshow_alpha(ax, t_datetime, f, data, alpha=a, cmap=cmap,
                                      vmin=rmin, vmax=rmax)
    
                if tr_Z == st_Z[0]:
                    cax, kw = make_axes(ax, location='left', fraction=0.07,
                                        pad=0.09) #pad=0.07
                    plt.colorbar(cm, cax=cax, ticks=xticks, **kw)
    
                for i in range(len(f)):
                    binned_data_signal_P[irow, i, :] += np.histogram(data[i,bol_signal_P_mask], bins=nbins,
                                                            range=(rmin, rmax),
                                                            weights=alpha[i,bol_signal_P_mask], density=True)[0]
                    binned_data_signal_S[irow, i, :] += np.histogram(data[i,bol_signal_S_mask], bins=nbins,
                                                            range=(rmin, rmax),
                                                            weights=alpha[i,bol_signal_S_mask], density=True)[0]
                    binned_data_noise[irow, i, :] += np.histogram(data[i,bol_noise_mask], bins=nbins,
                                                            range=(rmin, rmax),
                                                            weights=alpha[i,bol_noise_mask], density=True)[0]
                    
                    # histogram_data_signal_P[irow, i, :] += np.histogram(data[i,bol_signal_P_mask], bins=nbins,
                    #                                         range=(rmin, rmax),
                    #                                         weights=alpha[i,bol_signal_P_mask])[0]
                    # histogram_data_signal_S[irow, i, :] += np.histogram(data[i,bol_signal_S_mask], bins=nbins,
                    #                                         range=(rmin, rmax),
                    #                                         weights=alpha[i,bol_signal_S_mask])[0]
    
        date_fmt = mdates.DateFormatter('%Y-%m-%d \n %H:%M') #set time format: YYYY-MM-DD \n HH:MM in UTC
        loc = mdates.AutoDateLocator(tz=None, minticks=4, maxticks=6)
    
        
        for ax in axes:
            ax[0].set_xlim(utct(tstart).datetime, utct(tend).datetime)
            ax[0].xaxis.set_major_formatter(date_fmt)
            ax[0].xaxis.set_major_locator(loc)
            
            for a in ax[:-1]:
                a.set_ylim(fmin, fmax)
                a.set_ylabel("frequency / Hz")
            if log:
                ax[0].set_yscale('log')
            ax[0].yaxis.set_ticks_position('both')
            ax[1].yaxis.set_ticks_position('both')
            ax[2].yaxis.set_ticks_position('both')
            # set tick position twice, otherwise labels appear right :/
            # ax[signal_S_row].yaxis.set_ticks_position('right')
            # ax[signal_S_row].yaxis.set_label_position('right')
            ax[signal_S_row].yaxis.set_ticks_position('both')
            
            ax[density_row].yaxis.set_label_position('right')
            ax[density_row].yaxis.set_ticks_position('both')
    
        for ax in axes[0:-1, :].flatten():
            ax.set_xlabel('')
    
        for ax in axes[0:-1, 0]:
            ax.get_shared_x_axes().join(ax, axes[-1, 0])
            ax.get_shared_y_axes().join(ax, axes[-1, 0])
    
        for ax in axes[:, 1]:
            ax.set_ylabel('')
            ax.get_shared_y_axes().join(ax, axes[-1, 0])
            
        for ax in axes[:, 2]:
            ax.set_ylabel('')
            ax.get_shared_y_axes().join(ax, axes[-1, 0])
            
        for ax in axes[:, 3]:
            ax.set_ylabel('')
            ax.get_shared_y_axes().join(ax, axes[-1, 0])

    
        for i,ax in enumerate(axes[:, 0]):
            ax.grid(b=True, which='major', axis='x')
            
            #Patched marking the hist time windows
            ax.add_patch(rect[i][0])
            ax.add_patch(rect[i][1])
            ax.add_patch(rect[i][-1])
            
            #mark P/S arrival
            ax.axvline(x=utct(self.picks[phase_P]).datetime,ls='dashed',c='black')
            ax.axvline(x=utct(self.picks[phase_S]).datetime,ls='dashed',c='black')
            
    
        for ax in axes[0:-1, 0]:
            ax.set_xticklabels('')
            
            
        axes[0, signal_P_row].set_title('Signal P')
        axes[0, signal_S_row].set_title('Signal S')
        axes[0, noise_row].set_title('Noise')
        axes[0, density_row].set_title(f'Density \n {f_band_density[0]}-{f_band_density[1]} Hz')
        axes[0, 0].text(utct(tstart_signal_P).datetime, fmax+1, 'Signal P', c=color_windows[0], fontsize=12)
        axes[0, 0].text(utct(tstart_signal_S).datetime, fmax+1, 'Signal S', c=color_windows[1], fontsize=12)
        axes[0, 0].text(utct(tstart_noise).datetime, fmax+1, 'Noise', c=color_windows[-1], fontsize=12)
        axes[0, 0].text(utct(self.picks[phase_P]).datetime, fmin-0.3*fmin, phase_P, c='black', fontsize=12)
        axes[0, 0].text(utct(self.picks[phase_S]).datetime, fmin-0.3*fmin, phase_S, c='black', fontsize=12)
        
    
        linewidth_twofour = 1.0
        for irow, [data, rmin, rmax, a, xlabel, xticks, cmap] in \
                enumerate(iterables):
            
            #hist plot: signal P
            ax = axes[irow, signal_P_row]
            cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                               f, binned_data_signal_P[irow] *(rmax-rmin),
                               cmap='hot_r', #pqlx,
                               vmin=0., vmax=10,
                               shading='auto')
            
            # #Mark 2.4Hz and the BAZ of the event (if known)
            # ax.axhline(y=2.0,c='black', lw=linewidth_twofour)
            # ax.axhline(y=2.8,c='black', lw=linewidth_twofour)
            # if irow == 1 and self.baz:
            #     ax.axvline(x=self.baz,ls='dashed',c='darkgrey')
            ax.set_ylim(fmin, fmax)
            ax.set_xticks(xticks)
            
            #hist plot: signal S
            ax = axes[irow, signal_S_row]
            cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                               f, binned_data_signal_S[irow] *(rmax-rmin),
                               cmap='hot_r', #pqlx,
                               vmin=0., vmax=10,
                               shading='auto')
            
            # #Mark 2.4Hz and the BAZ of the event (if known)
            # ax.axhline(y=2.0,c='black', lw=linewidth_twofour)
            # ax.axhline(y=2.8,c='black', lw=linewidth_twofour)
            # if irow == 1 and self.baz:
            #     ax.axvline(x=self.baz,ls='dashed',c='darkgrey')
            ax.set_ylim(fmin, fmax)
            ax.set_xticks(xticks)
            
            #hist plot: noise
            ax = axes[irow, noise_row]
            cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                               f, binned_data_noise[irow] *(rmax-rmin),
                               cmap='hot_r', #pqlx,
                               vmin=0., vmax=10,
                               shading='auto')
            
            # #Mark 2.4Hz
            # ax.axhline(y=2.0,c='black', lw=linewidth_twofour)
            # ax.axhline(y=2.8,c='black', lw=linewidth_twofour)
                
            ax.set_ylim(fmin, fmax)
            if log:
                for i in range(0, 4):
                    axes[irow, i].set_yscale('log')
                    axes[irow, i].set_yticks((0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
                    axes[irow, i].set_yticklabels(("1/10", "1/5", "1/2", "1", "2", "5", "10"))
                    axes[irow, i].yaxis.set_minor_formatter(NullFormatter()) #removes minor ticks between the major ticks which are set above
                    axes[irow, i].set_ylim(fmin, fmax)
            ax.set_xticks(xticks)
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            
            ax = axes[irow, 0]
            ax.text(x=-0.25, y=0.5, transform=ax.transAxes, s=xlabel, #x=-0.18 #x=-019 gze
                    ma='center', va='center', bbox=props, rotation=90, size=10)
            
            # for ax in axes[1:, 1:-1].flatten():
            #     ax.grid(b=True, which='both', axis='x', linewidth=0.2, color='grey')
               
                
           #density curves over some frequency band
            hx_P = np.nansum(binned_data_signal_P[irow,idx_density[0],:], axis=0)
            hx_S = np.nansum(binned_data_signal_S[irow,idx_density[0],:], axis=0)
            hx_noise = np.nansum(binned_data_noise[irow,idx_density[0],:], axis=0)
            ax = axes[irow, density_row]
            # sns.kdeplot(x=hx_P, ax=ax, color=color_windows[0])
            # sns.kdeplot(x=hx_S, ax=ax, color=color_windows[1])
            ax.fill_between(np.linspace(rmin, rmax, nbins), hx_noise, 0,
                 facecolor="grey", # The fill color
                 color='grey',       # The outline color
                 alpha=0.2)          # Transparency of the fill
            ax.plot(np.linspace(rmin, rmax, nbins),hx_P, color=color_windows[0])
            ax.plot(np.linspace(rmin, rmax, nbins),hx_S, color=color_windows[1])
            # ax.plot(np.linspace(rmin, rmax, nbins),hx_noise, color=color_windows[-1])
            
            ax.set_xticks(xticks)
            ax.set_ylim(0,max(max(hx_P),max(hx_S))+0.05*max(max(hx_P),max(hx_S)))
            ax.set_xlim(rmin,rmax)
        
        
        #Mark the 2.4Hz band, set grid lines, mark BAZ
        for ax in axes[1, 1:-1]:
            if self.baz:
                ax.axvline(x=self.baz,ls='dashed',c='darkgrey')
        for ax in axes[:, 1:-1].flatten():
            ax.axhline(y=2.0,c='black', lw=linewidth_twofour)
            ax.axhline(y=2.8,c='black', lw=linewidth_twofour)
        for ax in axes[1:, 1:-1].flatten():
            ax.grid(b=True, which='both', axis='x', linewidth=0.2, color='grey')
            
    
        cbar_axes = fig.add_axes([gridspec_kw['right'] + dx_cbar,
                                  gridspec_kw['bottom'], w_cbar,
                                  gridspec_kw['top'] - gridspec_kw['bottom']])
        plt.colorbar(cm, cax=cbar_axes, label='weighted relative frequency')
        
        #For BSSA plot
        # axes[0,0].text(-0.21, 1.06, '(a)', fontsize=14, transform=axes[0,0].transAxes)
        # axes[0,1].text(-0.18, 1.06, '(b)', fontsize=14, transform=axes[0,1].transAxes)
        # axes[0,2].text(-0.18, 1.06, '(c)', fontsize=14, transform=axes[0,2].transAxes)
        # axes[1,0].text(-0.21, 1.06, '(d)', fontsize=14, transform=axes[1,0].transAxes)
        # axes[1,1].text(-0.18, 1.06, '(e)', fontsize=14, transform=axes[1,1].transAxes)
        # axes[1,2].text(-0.18, 1.06, '(f)', fontsize=14, transform=axes[1,2].transAxes)
    
        fig.suptitle(title)
    
        if fname is None:
            plt.show()
        else:
            savename = f'{fname}_diff' if differentiate else f'{fname}'
            # fig.savefig(f'Plots/Polarisation/{savename}.png', dpi=200) 
            if impact:
                path = f'Plots/Impact_search/Impact_{impact}'
            else:
                path = 'Plots/Test'
            fig.savefig(f'{path}/{savename}.png', dpi=200) if plot_6C or plot_spec_azi_only else fig.savefig(f'{path}/{savename}_4panels.png', dpi=200)