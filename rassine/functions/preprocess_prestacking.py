from pathlib import Path

import numpy as np

from ..io import open_pickle


def preprocess_prestacking(files_to_process, bin_length=1, dbin=0):
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

    Returns
    -------

    """
    files_to_process = np.sort(files_to_process)
    directory = Path(files_to_process[0]).parent.parent / "STACKED"

    directory.mkdir(parents=True, exist_ok=True)

    print("Loading the data, wait... \n")
    jdb = []
    berv = []
    lamp = []
    plx = []
    acc_sec = []

    for name in files_to_process:
        data = open_pickle(name)
        jdb.append(data["jdb"])
        berv.append(data["berv"])
        lamp.append(data["lamp_offset"])
        plx.append(data["plx_mas"])
        acc_sec.append(data["acc_sec"])

    jdb = np.array(jdb) + dbin
    berv = np.array(berv)
    lamp = np.array(lamp)
    plx = np.array(plx)
    acc_sec = np.array(acc_sec)

    if bin_length == 0:
        group = np.arange(len(jdb))
        groups = np.arange(len(jdb))
    else:
        groups = (jdb // bin_length).astype("int")
        groups -= groups[0]
        group = np.unique(groups)

    print("Number of bins : %.0f" % (len(group)))

    return jdb, berv, lamp, acc_sec, groups
