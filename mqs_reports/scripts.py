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
import obspy

from mqs_reports.catalog import Catalog
from mqs_reports.noise import read_noise


def fig_noise_stats():
    noise = read_noise('noise_0301_1024.npz')

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    noise.plot_noise_stats(ax=ax[0], show=False)

    cat = Catalog(fnam_quakeml='mqs_reports/data/catalog_20191024.xml',
                  quality=['A', 'B', 'C', 'D'])
    inv = obspy.read_inventory('mqs_reports/data/inventory.xml')
    sc3_path = '/mnt/mnt_sc3data'
    cat.load_distances(fnam_csv='./mqs_reports/data/manual_distances.csv')
    cat.read_waveforms(inv=inv, sc3dir=sc3_path)
    cat.calc_spectra(winlen_sec=10.)

    amps_HF = [event.amplitudes['A_24'] - 3 for event
            in cat.select(event_type=['24', 'HF'])]
    amps_LF = []
    for event in cat.select(event_type=['LF', 'BB']):
        amp_P = event.pick_amplitude(
            pick='Peak_MbP',
            comp='vertical',
            fmin=1. / 6.,
            fmax=1. / 1.5,
            instrument='VBB'
            )
        amp_S = event.pick_amplitude(
            pick='Peak_MbS',
            comp='vertical',
            fmin=1. / 6.,
            fmax=1. / 1.5,
            instrument='VBB'
            )
        amp = max((amp_P, amp_S))
        if amp is not None:
            amps_LF.append(20 * np.log10(amp))
    bins= np.arange(-240, -120, 5)
    ax[1].hist([amps_LF, amps_HF], bins)  # , labels=['LF events', 'HF events'])
    ax[0].legend()

    ax[1].set_xlabel('displacement PSD [dB]')
    ax[0].set_ylabel('percentage of mission')
    ax[1].set_ylabel('number of events')
    ax[1].set_xlim(-240, -160)
    plt.tight_layout()
    plt.savefig('noise_event_distribution.pdf')
    plt.show()


def get_mag_dist_prob(dists, mags):
    from mqs_reports.magnitudes import mb_S, M2_4
    noise = read_noise('noise_0301_1024.npz')
    power, p_LF, p_HF = noise.calc_noise_stats()

    funx = [mb_S, M2_4]
    p_ipl = np.zeros((2, len(dists), len(mags[0])))
    for i, p in enumerate((p_LF, p_HF)):
        for idist, dist in enumerate(dists):
            p_ipl[i, idist, :] = np.interp(x=mags[i],
                                           xp=funx[i](amplitude_dB=power,
                                                      distance_degree=dist),
                                           fp=p
                                           )

    return p_ipl


def fig_probs():
    # power = np.asarray(10**(power_dB / 20.))
    dists = np.arange(1, 181, 2)
    mags = [np.arange(2.5, 4.55, 0.05),
            np.arange(1.5, 3.55, 0.05)]
    distmaxs = [180, 45]
    fig, ax = plt.subplots(2, 1, figsize=(6, 9))
    p_ipl = get_mag_dist_prob(dists, mags)

    np.savez(file='probs.npz', p=p_ipl, dists=dists, mags=mags)

    for i in range(0, 2):
        distmax = distmaxs[i]
        ax[i].contourf(dists[dists < distmax], mags[i],
                       p_ipl[i, dists<distmax, :].T,
                       levels=[-0.5, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
                          cmap='Greys_r')
        cs = ax[i].contour(dists[dists < distmax], mags[i],
                           p_ipl[i, dists < distmax, :].T,
                           levels=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        ax[i].clabel(cs, inline=1, fontsize=10, colors='orange')

    cat = Catalog(fnam_quakeml='mqs_reports/data/catalog_20191024.xml',
                  quality=['A', 'B', 'C', 'D'])
    inv = obspy.read_inventory('mqs_reports/data/inventory.xml')
    sc3_path = '/mnt/mnt_sc3data'
    cat.read_waveforms(inv=inv, sc3dir=sc3_path)
    cat.calc_spectra(winlen_sec=10.)
    cat.load_distances(fnam_csv='./mqs_reports/data/manual_distances.csv')
    mags_LF = []
    mags_HF = []
    for event in cat.select(event_type=['LF', 'BB']):
        if event.distance is not None:

            # magnitude = event.magnitude(mag_type='MFB',
            # distance=event.distance)
            magnitude = event.magnitude(mag_type='mb_S',
                                        distance=event.distance)

            ax[0].scatter(event.distance,
                          magnitude,
                          color='k')
            ax[0].text(event.distance,
                       magnitude,
                    s=event.name, rotation=45,
                    verticalalignment='bottom',
                    horizontalalignment='left')
            mags_LF.append(magnitude)
    for event in cat.select(event_type=['HF', '24']):
        if event.distance is not None:
            mag = event.magnitude(mag_type='m2.4',
                                  distance=event.distance)
            ax[1].scatter(event.distance, mag,
                       color='k')
            mags_HF.append(mag)

    ax[0].set_xlim(0, 180)
    ax[1].set_xlim(0, 40)
    ax[0].set_title('Low-frequency events')
    ax[1].set_title('High-frequency events')
    for i in range(0, 2):
        ax[i].set_ylim(mags[i][0], mags[i][-1])
    ax[0].set_xlabel('distance / degrees')
    ax[1].set_xlabel('distance / degrees')
    ax[0].set_ylabel('magnitude M$_W$ (from S-picks)')
    ax[1].set_ylabel('magnitude M$_W$ (from 2.4 Hz peak)')
    plt.tight_layout()
    np.savetxt('mags_noncum_LF.txt', X=mags_LF)
    np.savetxt('mags_noncum_HF.txt', X=mags_HF)
    fig.savefig('events_detection_probability.pdf')

    plt.show()


    pass

if __name__=='__main__':
    fig_probs()
    fig_noise_stats()