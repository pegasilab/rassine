from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic.dataclasses import dataclass as pdataclass

from .tybles import Row, Table

# Columns

# fileroot and filename (virtual): complete path, is transformed in a new filename column which contains only the "name" without folders

# mjd: Needed for observation time

# TODO: there is duplicate information between DACE and FITS? lamp? what do we do?


@pdataclass(frozen=True)  # we use pydantic dataclasses to get validation
class Basic(Row):
    """
    Format of the rows in the meta table

    """

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Name only of the file itself, excluding folders
    filename: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction
    model: Optional[np.float64]

    @classmethod
    def after_read_(cls, table: pd.DataFrame) -> pd.DataFrame:
        if "filename" not in table.columns:
            table["filename"] = [str(Path(f).name) for f in table["fileroot"]]
        return table

    @classmethod
    def before_write_(cls, table: pd.DataFrame) -> pd.DataFrame:
        table.drop("filename")
        return table


@pdataclass(frozen=True)
class BorderScanned(Basic):
    berv: np.float64
    lamp: np.float64
    plx_mas: np.float64
    #: RV shift correction (diff from rv_mean)
    rv: np.float64
    # wave_min
    # wave_max
    pass


@pdataclass(frozen=True)
class BorderScannedGeneral:

    # TODO: document fields

    wave_min_k: np.float64
