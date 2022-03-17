import pickle
from pathlib import Path

import numpy.typing as npt
import pandas as pd
from astropy.io import fits

default_pickle_protocol: int = 3


def read_rv(file: Path) -> npt.ArrayLike:
    """
    Reads the RV values

    TODO: document this

    Parameters
    ----------
    file

    Returns
    -------
    Read values
    """

    if file.suffix == ".csv":
        # RV time-series to remove in kms, (binary  or known planets) otherwise give the systemic velocity
        rv = pd.read_csv(file)["model"]
    elif file.suffix == ".p":
        rv = pd.read_pickle(file)["model"]
    else:
        raise ValueError("Cannot read this file format")

    return rv


def open_pickle(filename):
    if filename.split(".")[-1] == "p":
        a = pd.read_pickle(filename)
        return a
    elif filename.split(".")[-1] == "fits":
        data = fits.getdata(filename)
        header = fits.getheader(filename)
        return data, header


def save_pickle(filename: str, output: dict, protocol: int = default_pickle_protocol):

    """
    Save a pickle file with the proper protocol pickle version.

    Parameters
    ----------
    filename : str
        Name of the output pickle file.
    output : dict
        Output dictionnary table to save.

    Returns
    -------

    """
    if filename.split(".")[-1] == "p":
        pickle.dump(output, open(filename, "wb"), protocol=protocol)
    if filename.split(".")[-1] == "fits":  # TODO: for future work
        pass
