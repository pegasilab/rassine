import glob as glob
import multiprocessing as multicpu
import os
import pickle
import sys
import time
import typing
from itertools import repeat

import matplotlib.pylab as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from colorama import Fore
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import norm

from ..analysis import clustering, find_nearest1, grouping, match_nearest, rolling_iq
from ..io import open_pickle, save_pickle
from ..math import create_grid, doppler_r, gaussian


def matching_diff_continuum(
    names, sub_dico="matching_anchors", master=None, savgol_window=200, zero_point=False
):
    """
    Match the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference.

    Parameters
    ----------
    names : array_like
        List of RASSINE files.
    sub_dico : str
        Name of the continuum to use. Either 'output' (RASSINE individual) or 'matching_anchors' (RASSINE time-series)
    master : str
        Name of the RASSINE master spectrum file.
    savgol window : int
        Length of the window for the savgol filtering.
    zero_point : bool
        No more used ?

    Returns
    -------

    """

    snr = []
    for j in names:
        file = open_pickle(j)
        snr.append(file["parameters"]["SNR_5500"])

    names = np.array(names)[np.array(snr).argsort()[::-1]]
    snr = np.array(snr)[np.array(snr).argsort()[::-1]]

    if master is None:
        master = names[0]

    master_name = master.split("/")[-1]
    file_highest = open_pickle(master)
    dx = np.diff(file_highest["wave"])[0]
    length_clip = int(100 / dx)  # smoothing on 100 \ang for the tellurics clean

    keys = file_highest[sub_dico].keys()

    all_continuum = [
        "continuum_linear",
        "continuum_cubic",
        "continuum_linear_denoised",
        "continuum_cubic_denoised",
    ]
    continuum_to_reduce = []
    for i in all_continuum:
        if i in keys:
            continuum_to_reduce.append(i)

    for i, j in enumerate(names):
        valid = True
        file = open_pickle(j)
        try:
            valid = file["matching_diff"]["parameters"]["reference_continuum"] != master_name
        except:
            pass

        if valid:
            print(
                "Modification of file (%.0f/%.0f) : %s (SNR : %.0f)"
                % (i + 1, len(names), j, snr[i])
            )
            spectre = file["flux_used"]

            par = {
                "reference_continuum": master_name,
                "savgol_window": savgol_window,
                "recenter": zero_point,
                "sub_dico_used": sub_dico,
            }
            file["matching_diff"] = {"parameters": par}

            for label in continuum_to_reduce:
                cont = file[sub_dico][label]
                cont2 = file_highest["flux_used"] / file_highest[sub_dico][label]

                cont1 = spectre / cont
                diff = cont1 - cont2
                med_value = np.nanmedian(diff)
                for k in range(3):  # needed to avoid artefact induced by tellurics
                    q1, q3, iq = rolling_iq(diff, window=length_clip)
                    diff[(diff > q3 + 1.5 * iq) | (diff < q1 - 1.5 * iq)] = med_value
                    diff[diff < q1 - 1.5 * iq] = med_value

                correction = smooth(diff, savgol_window, shape="savgol")
                correction = smooth(correction, savgol_window, shape="savgol")
                if zero_point:
                    correction = correction - np.nanmedian(correction)
                cont_corr = cont.copy() / (1 - correction.copy())
                file["matching_diff"][label] = cont_corr

            save_pickle(j, file)
