from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic.dataclasses import dataclass as pdataclass

# Columns

# fileroot and filename (virtual): complete path, is transformed in a new filename column which contains only the "name" without folders

# mjd: Needed for observation time

@pdataclass(frozen=True)
class MetaTableRow(TypedDict):
    """
    Format of the rows in the meta table
    """

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Name only of the file itself, excluding folders
    filename: str

    #: Observation date/time in MJD
    mjd: np.float64


@dataclass(frozen=True)
class MetaTable:
    table: pd.DataFrame

    @staticmethod
    def read_csv(csv_file: Path) -> MetaTable:
        """
        Reads a DACE CSV file which represents a MetaTable

        Adds a "filename" column with only the filename

        Args:
            csv_file: Path of file to read

        Returns:
            The enriched meta table
        """
        table = pd.read_csv(csv_file)
        if "filename" not in table.columns:
            table["filename"] = [str(Path(f).name) for f in table["fileroot"]]
        return MetaTable(table)

    def __getitem__(self, i: int) -> MetaTableRow:
        """
        Retrieves a row from
        Args:
            i:

        Returns:

        """
        return MetaTableRow(**self.table.iloc[i].to_dict())  # type: ignore


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
