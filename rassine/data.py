from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt


class MetaTable(TypedDict):
    """
    Format of the rows in the meta table
    """

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Name only of the file itself, excluding folders
    filename: str

    #: Observation date/time in MJD
    mjd: np.float64


class Preprocessed(TypedDict):
    """
    Data format of the pickle files produced by the preprocessing step
    """

    #: TODO: doc
    wave: Optional[npt.NDArray[np.float64]]
    #: TODO: doc, size=spectrum length
    flux: npt.NDArray[np.float64]
    #: TODO: doc, size=spectrum length
    flux_err: npt.NDArray[np.float64]
    #: instrument name
    instrument: str
    #: observation time in mjd
    mjd: np.float64
    #: what is jdb?
    jdb: np.float64
    #: what is berv?
    berv: np.float64
    #: what is lamp offset?
    lamp_offset: np.float64
    #: what is plx_mas?
    plx_mas: np.float64
    #: what is acc_sec?
    acc_sec: int
    #: what is wave_min?
    wave_min: np.float64
    #: what is wave_max?
    wave_max: np.float64
    #: what is dwave?
    dwave: np.float64
