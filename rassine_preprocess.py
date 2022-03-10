#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:41:21 2019

@author: cretignier
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from colorama import Fore

from rassine.functions import save_pickle


def preprocess_fits(
    files_to_process, instrument="HARPS", plx_mas=0, final_sound=True, output_dir=None
):
    """
    Preprocess the files spectra to produce readeable RASSINE files.

    The preprocessing depends on the s1d format of the instrument : HARPS, HARPN, CORALIE or ESPRESSO (if new DRS is used, use the ESPRESSO kw not matter the instrument)

    Parameters
    ----------
    files_to_process : array_like
        List of s1d spectra .fits spectra.
    instrument : str
        Instrument format of the s1d spectra.
    plx_mas : float
        parallaxe in mas (no more necessary ?)
    final_sound : bool
        Produce a sound once the preprocessing has finished (MacOS)
    output_dir : str
        Name of the output directory. If None, the output directory is created at the same location than the spectra.

    Returns
    -------

    """

    files_to_process = np.sort(files_to_process)
    number_of_files = len(files_to_process)
    counter = 0
    init_time = time.time()

    if (instrument == "HARPS") | (instrument == "CORALIE") | (instrument == "HARPN"):

        if output_dir is not None:
            directory = output_dir

        directory0, dustbin = os.path.split(files_to_process[0])
        if output_dir is not None:
            directory0 = output_dir
        try:
            table = pd.read_csv(directory0 + "/DACE_TABLE/Dace_extracted_table.csv")
            print("[INFO] DACE table has been found. Reduction will run with it.")
        except FileNotFoundError:
            print("[INFO] The DACE table is missing. Reduction will run without it.")

        for spectrum_name in files_to_process:

            counter += 1
            if (counter + 1) % 100 == 1:
                after_time = time.time()
                time_it = (after_time - init_time) / 100
                init_time = after_time
                if time_it > 1:
                    print(
                        "[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f s/it, remaining time : %.0f min %s s)"
                        % (
                            counter,
                            number_of_files,
                            time_it,
                            ((number_of_files - counter) * time_it) // 60,
                            str(int(((number_of_files - counter) * time_it) % 60)).zfill(2),
                        )
                    )
                else:
                    print(
                        "[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f it/s, remaining time : %.0f min %s s)"
                        % (
                            counter,
                            number_of_files,
                            time_it**-1,
                            ((number_of_files - counter) * time_it) // 60,
                            str(int(((number_of_files - counter) * time_it) % 60)).zfill(2),
                        )
                    )

            directory, name = os.path.split(spectrum_name)
            if output_dir is not None:
                directory = output_dir

            if not os.path.exists(directory + "/PREPROCESSED/"):
                os.system("mkdir " + directory + "/PREPROCESSED/")

            try:
                header = fits.getheader(spectrum_name)  # load the fits header
            except:
                print(
                    Fore.YELLOW
                    + "\n [WARNING] File %s cannot be read" % (spectrum_name)
                    + Fore.WHITE
                )
            spectre = fits.getdata(spectrum_name).astype("float64")  # the flux of your spectrum
            spectre_step = np.round(fits.getheader(spectrum_name)["CDELT1"], 8)
            wave_min = np.round(header["CRVAL1"], 8)  # to round float32
            wave_max = np.round(
                header["CRVAL1"] + (len(spectre) - 1) * spectre_step, 8
            )  # to round float32

            grid = np.round(np.linspace(wave_min, wave_max, len(spectre)), 8)

            begin = np.min(np.arange(len(spectre))[spectre > 0])
            end = np.max(np.arange(len(spectre))[spectre > 0])
            grid = grid[begin : end + 1]
            spectre = spectre[begin : end + 1]
            wave_min = np.min(grid)
            wave_max = np.max(grid)

            kw = "ESO"
            if instrument == "HARPN":
                kw = "TNG"

            berv = header["HIERARCH " + kw + " DRS BERV"]
            lamp = header["HIERARCH " + kw + " DRS CAL TH LAMP OFFSET"]
            try:
                pma = header["HIERARCH " + kw + " TEL TARG PMA"] * 1000
                pmd = header["HIERARCH " + kw + " TEL TARG PMD"] * 1000
            except:
                pma = 0
                pmd = 0

            if plx_mas:
                distance_m = 1000.0 / plx_mas * 3.08567758e16
                mu_radps = (
                    np.sqrt(pma**2 + pmd**2)
                    * 2
                    * np.pi
                    / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
                )
                acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
            else:
                acc_sec = 0

            if instrument == "CORALIE":
                if np.mean(spectre) < 100000:
                    spectre *= (
                        400780143771.18976  # calibrated with HD8651 2016-12-16 AND 2013-10-24
                    )

                spectre /= 1.4e10 / 125**2  # calibrated to match with HARPS SNR

            try:
                mjd = table.loc[table["fileroot"] == spectrum_name, "mjd"].values[0]
            except NameError:
                try:
                    mjd = header["MJD-OBS"]
                except KeyError:
                    mjd = Time(name.split("/")[-1].split(".")[1]).mjd

            jdb = np.array(mjd) + 0.5

            out = {
                "flux": spectre,
                "flux_err": np.zeros(len(spectre)),
                "instrument": instrument,
                "mjd": mjd,
                "jdb": jdb,
                "berv": berv,
                "lamp_offset": lamp,
                "plx_mas": plx_mas,
                "acc_sec": acc_sec,
                "wave_min": wave_min,
                "wave_max": wave_max,
                "dwave": spectre_step,
            }

            save_pickle(directory + "/PREPROCESSED/" + name[:-5] + ".p", out)

    elif (instrument == "ESPRESSO") | (instrument == "EXPRESS"):

        if output_dir is not None:
            directory = output_dir

        directory0, dustbin = os.path.split(files_to_process[0])
        if output_dir is not None:
            directory0 = output_dir
        try:
            table = pd.read_csv(directory0 + "/DACE_TABLE/Dace_extracted_table.csv")
            print("[INFO] DACE table has been found. Reduction will run with it.")
        except FileNotFoundError:
            print("[INFO] The DACE table is missing. Reduction will run without it.")

        for spectrum_name in files_to_process:

            counter += 1
            if (counter + 1) % 100 == 1:
                after_time = time.time()
                time_it = (after_time - init_time) / 100
                init_time = after_time
                if time_it > 1:
                    print(
                        "[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f s/it, remaining time : %.0f min %s s)"
                        % (
                            counter,
                            number_of_files,
                            time_it,
                            ((number_of_files - counter) * time_it) // 60,
                            str(int(((number_of_files - counter) * time_it) % 60)).zfill(2),
                        )
                    )
                else:
                    print(
                        "[INFO] Number of files preprocessed : --- %.0f/%0.f --- (%.2f it/s, remaining time : %.0f min %s s)"
                        % (
                            counter,
                            number_of_files,
                            time_it**-1,
                            ((number_of_files - counter) * time_it) // 60,
                            str(int(((number_of_files - counter) * time_it) % 60)).zfill(2),
                        )
                    )

            directory, name = os.path.split(spectrum_name)
            if output_dir is not None:
                directory = output_dir

            if not os.path.exists(directory + "/PREPROCESSED/"):
                os.system("mkdir " + directory + "/PREPROCESSED/")

            header = fits.getheader(spectrum_name)  # load the fits header
            spectre = fits.getdata(spectrum_name)["flux"].astype(
                "float64"
            )  # the flux of your spectrum
            spectre_error = fits.getdata(spectrum_name)["error"].astype(
                "float64"
            )  # the flux of your spectrum
            grid = fits.getdata(spectrum_name)["wavelength_air"].astype(
                "float64"
            )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)
            begin = np.min(
                np.arange(len(spectre))[spectre > 0]
            )  # remove border spectrum with 0 value
            end = np.max(
                np.arange(len(spectre))[spectre > 0]
            )  # remove border spectrum with 0 value
            grid = grid[begin : end + 1]
            spectre = spectre[begin : end + 1]
            spectre_error = spectre_error[begin : end + 1]
            wave_min = np.min(grid)
            wave_max = np.max(grid)
            spectre_step = np.mean(np.diff(grid))
            try:
                mjd = table.loc[table["fileroot"] == spectrum_name, "mjd"].values[0]
            except NameError:
                try:
                    mjd = float(header["MJD-OBS"])
                except KeyError:
                    mjd = Time(name.split("/")[-1].split(".")[1]).mjd

            kw = "ESO"
            if "HIERARCH TNG QC BERV" in header:
                kw = "TNG"

            berv = float(header["HIERARCH " + kw + " QC BERV"])
            lamp = 0  # header['HIERARCH ESO DRS CAL TH LAMP OFFSET'] no yet available
            try:
                pma = header["HIERARCH " + kw + " TEL TARG PMA"] * 1000
                pmd = header["HIERARCH " + kw + " TEL TARG PMD"] * 1000
            except:
                pma = 0
                pmd = 0

            if plx_mas:
                distance_m = 1000.0 / plx_mas * 3.08567758e16
                mu_radps = (
                    np.sqrt(pma**2 + pmd**2)
                    * 2
                    * np.pi
                    / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
                )
                acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
            else:
                acc_sec = 0
            jdb = np.array(mjd) + 0.5

            out = {
                "wave": grid,
                "flux": spectre,
                "flux_err": spectre_error,
                "instrument": instrument,
                "mjd": mjd,
                "jdb": jdb,
                "berv": berv,
                "lamp_offset": lamp,
                "plx_mas": plx_mas,
                "acc_sec": acc_sec,
                "wave_min": wave_min,
                "wave_max": wave_max,
                "dwave": spectre_step,
            }

            save_pickle(directory + "/PREPROCESSED/" + name[:-5] + ".p", out)

    if final_sound:
        make_sound("Preprocessing files has finished")


def get_parser() -> argparse.ArgumentParser:
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(
        allow_abbrev=False,
        description="""\
    RASSINE preprocessing tool
    
    Preprocess the files spectra to produce readable RASSINE files.
        
    The preprocessing depends on the s1d format of the instrument : HARPS, HARPN, CORALIE or ESPRESSO 
    (if new DRS is used, use the ESPRESSO kw not matter the instrument)
    """,
    )

    res.add_argument("inputfiles", type=Path, nargs="+", help="Input files to process")
    res.add_argument(
        "--instrument",
        "-i",
        type=str,
        default="HARPS",
        help="Instrument format of the s1d spectra",
    )
    res.add_argument(
        "--plx-mas", type=float, default=0.0, help="parallaxe in mas (no more necessary ?)"
    )
    res.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="",
        help="Name of the output directory. If None, the output directory is created at the same location than the spectra.",
    )
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    inputfiles = args.inputfiles

    assert len(inputfiles) > 0, "At least one input file must be specified"
    output_dir = args.output_dir
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = inputfiles[0].parent

    preprocess_fits(
        list(map(str, inputfiles)),
        instrument=args.instrument,
        plx_mas=args.plx_mas,
        final_sound=False,
        output_dir=str(output_dir),
    )
