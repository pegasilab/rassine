import typing

import numpy as np

from rassine.functions.misc import smooth

from ..analysis import rolling_iq
from ..io import open_pickle, save_pickle


def matching_diff_continuum(
    names_: typing.Sequence[str],
    sub_dico: typing.Literal["output", "matching_anchors"] = "matching_anchors",
    master_: typing.Optional[str] = None,
    savgol_window: int = 200,
    zero_point: bool = False,
):
    """
    Match the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference.

    Args:
        names_: List of RASSINE files.
        sub_dico : Name of the continuum to use. Either 'output' (RASSINE individual) or 'matching_anchors' (RASSINE time-series)
        master_: Name of the RASSINE master spectrum file.
        savgol window: Length of the window for the savgol filtering.
        zero_point: No more used ?
    """
    names = np.array(names_)

    snr = []
    for j in names:
        file = open_pickle(j)
        snr.append(file["parameters"]["SNR_5500"])

    names = np.array(names)[np.array(snr).argsort()[::-1]]
    snr = np.array(snr)[np.array(snr).argsort()[::-1]]

    if master_ is None:
        master = names[0]
    else:
        master = master_

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
