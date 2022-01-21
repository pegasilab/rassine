from pathlib import Path
import numpy.typing as npt
import pandas as pd


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

    if file.suffix == '.csv':
        # RV time-series to remove in kms, (binary  or known planets) otherwise give the systemic velocity
        rv = pd.read_csv(file)['model']
    elif file.suffix == '.p':
        rv = pd.read_pickle(file)['model']
    else:
        raise ValueError('Cannot read this file format')

    return rv
