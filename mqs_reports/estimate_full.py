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
from tqdm import tqdm


def mc_test(mag, nreal, p_ipl):
    # A: Randomly propose an annual rate of this magnitude
    annual_rate = np.random.rand() * (nreal + 1) * 200

    # B: Randomly estimate real number of events this year, given the rate
    #    from A
    ntrue = np.random.poisson(lam=annual_rate)

    # C: distribute events over distance
    distances = np.pi - np.arcsin(np.random.rand((ntrue)) * 2. - 1.)

    # D: Calculate, which events were detected or not
    probs = np.random.rand((ntrue))
    try:
        npred = np.sum(
            probs < p_ipl(np.rad2deg(distances), mag))  # prob_with_dist(
        # print(p_ipl(np.rad2deg(distances), mag))
    except ValueError:
        npred = 0
    # distances,
    # mag=mag))

    return annual_rate, npred


if __name__ == '__main__':
    from scipy.interpolate import interp2d

    mags = np.arange(2.25, 4.5, 0.25)

    data = np.loadtxt('mags_noncum_LF.txt')
    n_LF, bin_LF = np.histogram(data, bins=mags + 0.1)

    # data = np.loadtxt('mags_noncum_HF.txt')
    # n_HF, bin_HF = np.histogram(data, bins=mags + 0.1)

    n = n_LF
    mags = mags[1:]

    with np.load(file='probs.npz') as prob_data:
        mags_p = prob_data['mags'][0]
        dists = prob_data['dists']
        p = prob_data['p'][0]
    p_ipl = interp2d(x=dists, y=mags_p, z=p.T, kind='linear')

    ndraws = int(1e5)
    rates = []
    means = []
    perc95 = []
    perc05 = []
    preds = np.zeros(ndraws)
    for mag, nmag in zip(mags, n):
        rates.append([])
        for i in tqdm(range(ndraws)):
            annual_rate, npred = mc_test(mag, nreal=nmag, p_ipl=p_ipl)
            preds[i] = npred
            if np.abs(npred - nmag) < 0.5:
                rates[-1].append(annual_rate)
        # plt.hist(preds)
        # plt.axvline(nmag)
        # plt.show()
        if len(rates[-1]) == 0:
            means.append(0.)
            perc05.append(0.)
            perc95.append(0.)
        else:
            means.append(np.median(rates[-1]))
            perc05.append(np.percentile(rates[-1], q=5))
            perc95.append(np.percentile(rates[-1], q=95))
    plt.plot(mags, means, lw=2, ls='solid', c='k', label='mean')
    plt.plot(mags, perc05, lw=2, ls='dashed', c='k', label='5%')
    plt.plot(mags, perc95, lw=2, ls='dashed', c='k', label='95%')
    plt.plot(mags, n, 'o')
    plt.legend()
    plt.ylim(0.1, 5000)
    plt.yscale('log')
    plt.xlabel('magnitude $M_W$')
    plt.ylabel('number of events of $M_W$, non-cumulative')
    plt.savefig('noncum_HF.png', dpi=200)
    plt.show()
