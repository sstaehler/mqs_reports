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

from mqs_reports.utils import create_fnam_event, read_data, calc_PSD


class Event:
    def __init__(self,
                 name: str,
                 publicid: str,
                 origin_publicid: str,
                 picks: dict,
                 quality: str,
                 mars_event_type: str):
        self.name=name.strip(),
        self.publicid = publicid
        self.origin_publicid = origin_publicid
        self.picks = picks
        self.quality = quality
        self.mars_event_type = mars_event_type
        self._waveforms_read = False
        self._spectra_available = False


    def read_waveforms(self, inv, kind, sc3dir,
                       filenam_VBB_HG='XB.ELYSE.02.?H?.D.2019.%03d',
                       filenam_SP_HG='XB.ELYSE.65.EH?.D.2019.%03d'
                       ):
        if not self.read_data_local():
            self.read_data_from_sc3dir(filenam_SP_HG, filenam_VBB_HG,
                                       inv, kind, sc3dir)
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
                self.waveforms_VBB = obspy.read(pjoin(waveform_path,
                                                      'waveforms_VBB.mseed'))
                SP_path = pjoin(waveform_path, 'waveforms_SP.mseed')
                if len(glob(SP_path)):
                    self.waveforms_SP = obspy.read(SP_path)
                else:
                    self.waveforms_SP = None
                success = True
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


    def read_data_from_sc3dir(self, filenam_SP_HG, filenam_VBB_HG,
                              inv, kind, sc3dir, tpre_SP=100, tpre_VBB=900.):
        self.kind = kind
        fnam_VBB, fnam_SP = create_fnam_event(
            filenam_VBB_HG=filenam_VBB_HG,
            filenam_SP_HG=filenam_SP_HG,
            sc3dir=sc3dir, time=self.picks['start'])
        self.picks = self.picks
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
        if len(glob(fnam_SP)) > 0:
            # Use SP waveforms only if 65.EH? exists, not otherwise (we
            # don't need 20sps SP data)
            self.waveforms_SP = read_data(fnam_SP, inv=inv, kind=kind,
                                             twin=[twin_start - tpre_SP,
                                                    twin_end + tpre_SP],
                                             fmin=0.5)
        else:
            self.waveforms_SP = None
        self.waveforms_VBB = read_data(fnam_VBB, inv=inv,
                                       kind=kind,
                                       twin=[twin_start - tpre_VBB,
                                             twin_end + tpre_VBB])


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
                # else:
                #     self['spectra'][variable]['p_' + chan] = p
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

        self._spectra_available = True

    def pick_amplitude(self, pick, comp, fmin, fmax, instrument='VBB'):
        if instrument=='VBB':
            st_work = self.waveforms_VBB.copy()
        else:
            st_work = self.waveforms_SP.copy()

        st_work.filter('bandpass', zerophase=True, freqmin=fmin, freqmax=fmax)
        # if not self.kind=='DISP':
        #     raise ValueError('Waveform must be displacement for amplitudes')
        if self.picks[pick]=='':
            return None
        else:
            tmin = utct(self.picks[pick]) - 10.
            tmax = utct(self.picks[pick]) - 10.
            st_work.trim(starttime=tmin, endtime=tmax)
            if comp in ['Z', 'N', 'E']:
                return abs(st_work.select(channel='??' + comp)[0].data).max()
            elif comp == 'all':
                amp_N = abs(st_work.select(channel='??N')[0].data).max()
                amp_E = abs(st_work.select(channel='??E')[0].data).max()
                amp_Z = abs(st_work.select(channel='??Z')[0].data).max()
                return max((amp_E, amp_N, amp_Z))
            elif comp == 'horizontal':
                amp_N = abs(st_work.select(channel='??N')[0].data).max()
                amp_E = abs(st_work.select(channel='??E')[0].data).max()
                return max((amp_E, amp_N))
            elif comp == 'vertical':
                return abs(st_work.select(channel='??Z')[0].data).max()
