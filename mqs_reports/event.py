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
from mars_tools.insight_time import solify
from obspy import UTCDateTime as utct
from obspy.geodetics.base import kilometers2degrees, gps2dist_azimuth

from mqs_reports.annotations import Annotations
from mqs_reports.magnitudes import fit_spectra
from mqs_reports.utils import create_fnam_event, read_data, calc_PSD, detick

RADIUS_MARS = 3389.5
CRUST_VP = 4.
CRUST_VS = 4. / 3. ** 0.5
LANDER_LAT = 4.5024
LANDER_LON = 135.6234

EVENT_TYPES_SHORT = {
    'SUPER_HIGH_FREQUENCY': 'SF',
    'VERY_HIGH_FREQUENCY': 'VF',
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
                        print('Found aligned distance %f for event %s' %
                              (self.distance, self.name))

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
                mag_type == 'MFB_HF'
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
        # plt.show()

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
            st_work = self.waveforms_SP.select(channel='??[ENZ]').copy()
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
            # if normwindow == 'P' and len(tstart_norm[normwindow]) == 0:
            #     normwindow = 'all'
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
                        maxfac = np.quantile(tr_norm.data, q=0.9)
                        offset = np.quantile(tr_norm.data, q=0.1)

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
