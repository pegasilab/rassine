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


def intersect_all_continuum(names, add_new=True):
    """
    Perform the intersection of the RASSINE files by using the anchor location saved in the master RASSINE spectrum.

    Parameters
    ----------
    names : array_like
        List of RASSINE files.
    add_new : bool
        Add anchor points that were not detected.

    Returns
    -------

    """

    print("Extraction of the new continua, wait... \n")

    names = np.sort(names)
    sub_dico = "output"

    directory, dustbin = os.path.split(names[0])
    directory = "/".join(directory.split("/")[0:-1])

    master_file = glob.glob(directory + "/MASTER/RASSINE_Master_spectrum*")[0]
    names = np.hstack([master_file, names])
    number_of_files = len(names)

    tool_file = glob.glob(directory + "/MASTER/Master_tool*")[0]
    tool_name = tool_file.split("/")[-1]
    tool = pd.read_pickle(tool_file)

    fraction = tool["fraction"]
    tolerance = tool["tolerance"]
    threshold = tool["threshold"]
    copies_master = tool["nb_copies_master"]
    master_spectrum = tool["master_filename"]
    border2 = tool["border"]
    cluster_center = (border2[:, 1] - border2[:, 0]) / 2 + border2[:, 0]
    curve = tool["curve"]

    for i, j in enumerate(names):
        valid = True
        file = open_pickle(j)
        try:
            valid = file["matching_anchors"]["parameters"]["master_tool"] != tool_name
        except:
            pass

        if valid:
            spectrei = file["flux"]
            spectre = file["flux_used"]
            grid = file["wave"]

            index = file[sub_dico]["anchor_index"]
            wave = file[sub_dico]["anchor_wave"]
            flux = file[sub_dico]["anchor_flux"]

            save = index.copy()

            diff = np.min([np.diff(save[1:]), np.diff(save[0:-1])], axis=0)
            diff = np.array([diff[0]] + list(diff) + [diff[-1]])
            diff = diff * float(fraction)
            diff = diff.astype("int")
            mask = np.zeros(len(grid))
            new = []
            for k in range(len(save)):
                new.append(save[k] + np.arange(-diff[k], diff[k]))
            new = np.unique(np.hstack(new))
            new = new[(new > 0) & (new < len(mask))]
            mask[new.astype("int")] = 1

            test = mask * curve

            mask_idx = test[save].astype("bool")
            mask_idx[0 : file["parameters"]["number of cut"]] = True
            mask_idx[-file["parameters"]["number of cut"] :] = True

            try:
                flux_denoised = file[sub_dico]["anchor_flux_denoised"]
            except:
                flux_denoised = flux

            save_before = len(index)

            index = index[mask_idx]
            wave = wave[mask_idx]
            flux = flux[mask_idx]
            flux_denoised = flux_denoised[mask_idx]

            save_after = len(index)

            index2 = (index >= border2[:, 0][:, np.newaxis]) & (
                index <= border2[:, 1][:, np.newaxis]
            )
            cluster_empty = np.where(np.sum(index2, axis=1) == 0)[0]
            cluster_twin = np.where(np.sum(index2, axis=1) > 1)[0]

            if len(cluster_twin) != 0:
                index3 = np.unique(index * index2[cluster_twin, :])[1:]
                # centers_with_twin = cluster_center[cluster_twin]

                index_kept = index3[
                    match_nearest(cluster_center, index3)[:, 1].astype("int")
                ]  # only twin closest to the cluster is kept
                index_to_suppress = np.setdiff1d(index3, index_kept)
                mask_idx = ~np.in1d(index, index_to_suppress)

                index = index[mask_idx]
                wave = wave[mask_idx]
                flux = flux[mask_idx]
                flux_denoised = flux_denoised[mask_idx]

            save_after_twin = len(index)

            if add_new:
                index_max, flux_max = local_max(spectre, file["parameters"]["vicinity_local_max"])

                new_max_index = []
                new_max_flux = []
                new_max_flux_denoised = []
                new_max_wave = []

                for k in cluster_empty:
                    kept = (index_max >= border2[k, 0]) & (index_max <= border2[k, 1])
                    if sum(kept) != 0:
                        maxi = flux_max[kept].argmax()
                        new_max_index.append((index_max[kept].astype("int"))[maxi])
                        new_max_flux.append((flux_max[kept])[maxi])
                        new_max_wave.append(grid[(index_max[kept].astype("int"))[maxi]])
                        new_max_flux_denoised.append(
                            np.mean(
                                spectre[
                                    (index_max[kept].astype("int"))[maxi]
                                    - int(file["parameters"]["denoising_dist"]) : (
                                        index_max[kept].astype("int")
                                    )[maxi]
                                    + int(file["parameters"]["denoising_dist"])
                                    + 1
                                ]
                            )
                        )

                new_max_index = np.array(new_max_index)
                new_max_flux = np.array(new_max_flux)
                new_max_flux_denoised = np.array(new_max_flux_denoised)
                new_max_wave = np.array(new_max_wave)

                index = np.hstack([index, new_max_index])
                wave = np.hstack([wave, new_max_wave])
                flux = np.hstack([flux, new_max_flux])
                flux_denoised = np.hstack([flux_denoised, new_max_flux_denoised])

            save_after_new = len(index)
            print(
                "\nModification of file (%.0f/%.0f): %s \n# of anchor before : %.0f \n# of anchor after out-of-cluster filtering : %0.f \n# of anchor after twin filtering : %.0f \n# of anchor after adding : %.0f"
                % (
                    i + 1,
                    number_of_files,
                    j,
                    save_before,
                    save_after,
                    save_after_twin,
                    save_after_new,
                )
            )

            continuum1, continuum3, continuum1_denoised, continuum3_denoised = make_continuum(
                wave,
                flux,
                flux_denoised,
                grid,
                spectrei,
                continuum_to_produce=[
                    file["parameters"]["continuum_interpolated_saved"],
                    file["parameters"]["continuum_denoised_saved"],
                ],
            )

            float_precision = file["parameters"]["float_precision"]
            if float_precision != "float64":
                flux = flux.astype(float_precision)
                wave = wave.astype(float_precision)
                continuum3 = continuum3.astype(float_precision)
                continuum1 = continuum1.astype(float_precision)
                continuum3_denoised = continuum3_denoised.astype(float_precision)
                continuum1_denoised = continuum1_denoised.astype(float_precision)
                flux_denoised = flux_denoised.astype(float_precision)
            index = index.astype("int")

            outputs_interpolation_saved = file["parameters"]["continuum_interpolated_saved"]
            outputs_denoising_saved = file["parameters"]["continuum_denoised_saved"]

            if (outputs_interpolation_saved == "linear") & (
                outputs_denoising_saved == "undenoised"
            ):
                output = {
                    "continuum_linear": continuum1,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "cubic") & (
                outputs_denoising_saved == "undenoised"
            ):
                output = {
                    "continuum_cubic": continuum3,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "linear") & (
                outputs_denoising_saved == "denoised"
            ):
                output = {
                    "continuum_linear": continuum1_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux_denoised,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "cubic") & (
                outputs_denoising_saved == "denoised"
            ):
                output = {
                    "continuum_cubic": continuum3_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux_denoised,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "all") & (outputs_denoising_saved == "denoised"):
                output = {
                    "continuum_cubic": continuum3_denoised,
                    "continuum_linear": continuum1_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux_denoised,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "all") & (
                outputs_denoising_saved == "undenoised"
            ):
                output = {
                    "continuum_cubic": continuum3,
                    "continuum_linear": continuum1,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "linear") & (outputs_denoising_saved == "all"):
                output = {
                    "continuum_linear": continuum1,
                    "continuum_linear_denoised": continuum1_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_flux_denoised": flux_denoised,
                    "anchor_index": index,
                }
            elif (outputs_interpolation_saved == "cubic") & (outputs_denoising_saved == "all"):
                output = {
                    "continuum_cubic": continuum3,
                    "continuum_cubic_denoised": continuum3_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_flux_denoised": flux_denoised,
                    "anchor_index": index,
                }
            else:
                output = {
                    "continuum_cubic": continuum3,
                    "continuum_linear": continuum1,
                    "continuum_cubic_denoised": continuum3_denoised,
                    "continuum_linear_denoised": continuum1_denoised,
                    "anchor_wave": wave,
                    "anchor_flux": flux,
                    "anchor_flux_denoised": flux_denoised,
                    "anchor_index": index,
                }

            file["matching_anchors"] = output
            file["matching_anchors"]["parameters"] = {
                "master_tool": tool_name,
                "master_filename": master_spectrum,
                "sub_dico_used": sub_dico,
                "nb_copies_master": copies_master,
                "threshold": threshold,
                "tolerance": tolerance,
                "fraction": fraction,
            }
            save_pickle(names[i], file)
