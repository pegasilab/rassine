from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic.dataclasses import dataclass as pdataclass

# Columns

# fileroot and filename (virtual): complete path, is transformed in a new filename column which contains only the "name" without folders

# mjd: Needed for observation time

# TODO: there is duplicate information between DACE and FITS? lamp? what do we do?


T = TypeVar("T", bound=RowType)


@pdataclass(frozen=True)  # we use pydantic dataclasses to get validation
class MetaTableRow:
    """
    Format of the rows in the meta table

    """

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Name only of the file itself, excluding folders
    filename: str

    #: Observation date/time in MJD
    mjd: np.float_

    #: Optional RV shift correction
    model: Optional[np.float_] = None

    test_not_there: Optional[str] = None


@dataclass(frozen=True)
class MetaTable:
    """
    Meta table for processed spectra

    Rows must be already sorted according to filenames
    """

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
        Retrieves a row from the table

        Args:
            i: Row index

        Returns:
            A row dataclass
        """
        data: Mapping[str, Any] = self.table.iloc[i].to_dict()
        fs = fields(MetaTableRow)
        fieldnames: Sequence[str] = [f.name for f in fs]
        return MetaTableRow(**{k: v for k, v in data.items() if k in fieldnames})  # type: ignore


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
