import os
import time
from pathlib import Path

import numpy as np
from astropy.time import Time

from ..analysis import find_nearest
from ..io import open_pickle, save_pickle
from ..math import create_grid
from .preprocess_prestacking import preprocess_prestacking


def preprocess_stack(files_to_process, bin_length=1, dbin=0, make_master=True):
    """
    Stack all the spectra according to a defined binning length.

    Parameters
    ----------
    files_to_process : array_like
        List of s1d .p preprocessed spectra (common wavelength grid).
    bin_length : int
        Length for the binning in days (nightly binning = 1).
    dbin : float
        Shift in days of the binning starting point (nightly = 0, daily = 0.5)
    make_master : bool
        Produce a master spectrum by stacking all the observations.

    Returns
    -------

    """

    files_to_process = np.sort(files_to_process)
    directory = Path(files_to_process[0]).parent.parent / "STACKED"

    jdb, berv, lamp, rv_sec, groups = preprocess_prestacking(
        files_to_process, bin_length=bin_length, dbin=dbin
    )

    group = np.unique(groups)

    num = -1

    all_snr = []
    all_stack = []
    all_berv = []
    for j in group:
        num += 1
        g = np.where(groups == j)[0]
        file_arbitrary = open_pickle(files_to_process[0])
        wave_min = file_arbitrary["wave_min"]
        wave_max = file_arbitrary["wave_max"]
        dwave = file_arbitrary["dwave"]
        grid = create_grid(wave_min, dwave, len(file_arbitrary["flux"]))
        RV_sys = file_arbitrary["RV_sys"]
        instrument = file_arbitrary["instrument"]
        hole_left = file_arbitrary["hole_left"]
        hole_right = file_arbitrary["hole_right"]
        acc_sec = file_arbitrary["acc_sec"]
        stack = 0
        stack_err = 0
        bolo = []
        rv_shift = []
        name_root_files = []
        for file in files_to_process[g]:
            f = open_pickle(file)
            flux = f["flux"]
            flux_err = f["flux_err"]
            rv_shift.append(f["RV_shift"])
            stack += flux
            stack_err += flux_err**2
            bolo.append(np.nansum(flux) / len(flux))
            name_root_files.append(file)

        bolo = np.array(bolo)
        rv_shift = np.array(rv_shift)
        wave_ref = int(find_nearest(grid, 5500)[0])
        continuum_5500 = np.nanpercentile(stack[wave_ref - 50 : wave_ref + 50], 95)
        SNR = np.sqrt(continuum_5500)
        all_snr.append(SNR)
        all_stack.append(stack)
        nb_spectra_stacked = len(g)
        # photometric average weighted by the bolometry
        jdb_w = np.sum((jdb[g] - dbin) * bolo) / np.sum(bolo)
        date_name = Time(jdb_w - 0.5, format="mjd").isot
        berv_w = np.sum(berv[g] * bolo) / np.sum(bolo)
        lamp_w = np.sum(lamp[g] * bolo) / np.sum(bolo)
        all_berv.append(berv_w)
        out = {
            "flux": stack,
            "flux_err": np.sqrt(abs(stack_err)),
            "jdb": jdb_w,  # mjd weighted average by bolo flux
            "mjd": jdb_w - 0.5,  # mjd weighted average by bolo flux
            "berv": berv_w,
            "lamp_offset": lamp_w,
            "acc_sec": acc_sec,
            "RV_shift": np.sum(rv_shift * bolo) / np.sum(bolo),
            "RV_sys": RV_sys,
            "SNR_5500": SNR,
            "hole_left": hole_left,
            "hole_right": hole_right,
            "wave_min": wave_min,
            "wave_max": wave_max,
            "dwave": dwave,
            "stacking_length": bin_length,
            "nb_spectra_stacked": nb_spectra_stacked,
            "arcfiles": name_root_files,
        }

        save_pickle(directory / f"Stacked_spectrum_bin_{bin_length}.{date_name}.p", out)

    all_snr = np.array(all_snr)

    print(
        "SNR 5500 statistic (Q1/Q2/Q3) : Q1 = %.0f / Q2 = %.0f / Q3 = %.0f"
        % (
            np.nanpercentile(all_snr, 25),
            np.nanpercentile(all_snr, 50),
            np.nanpercentile(all_snr, 75),
        )
    )

    master_name = None

    if make_master:
        all_berv = np.array(all_berv)
        stack = np.array(all_stack)
        stack = np.sum(stack, axis=0)
        stack[stack <= 0] = 0
        continuum_5500 = np.nanpercentile(stack[wave_ref - 50 : wave_ref + 50], 95)
        SNR = np.sqrt(continuum_5500)
        BERV = np.sum(all_berv * all_snr**2) / np.sum(all_snr**2)
        BERV_MIN = np.min(berv)
        BERV_MAX = np.max(berv)
        # plt.figure(figsize=(16,6))
        # plt.plot(grid,stack,color='k')

        master_name = "Master_spectrum_%s.p" % (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))

        save_pickle(
            directory / master_name,
            {
                "flux": stack,
                "master_spectrum": True,
                "RV_sys": RV_sys,
                "RV_shift": 0,
                "SNR_5500": SNR,
                "lamp_offset": 0,
                "acc_sec": acc_sec,
                "berv": BERV,
                "berv_min": BERV_MIN,
                "berv_max": BERV_MAX,
                "instrument": instrument,
                "mjd": 0,
                "jdb": 0,
                "hole_left": hole_left,
                "hole_right": hole_right,
                "wave_min": wave_min,
                "wave_max": wave_max,
                "dwave": dwave,
                "nb_spectra_stacked": len(files_to_process),
                "arcfiles": "none",
            },
        )
        # plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
        # plt.ylabel('Flux',fontsize=14)
        # plt.show()
        # loop = sphinx('Press Enter to finish the stacking process.')
        # plt.close()

    return master_name
