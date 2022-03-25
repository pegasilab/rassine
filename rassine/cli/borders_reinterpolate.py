import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from configpile import *
from typing_extensions import Annotated

from ..tybles import Row, Table
from .base import BasicInfo, RassineConfigBeforeStack, RelPath, RootPath

# Inputs:
# border_scan_output
# preprocessed6Matinno



@dataclass(frozen=True)
class Task(RassineConfigBeforeStack):
    """
    Reinterpolate the spectra according to the general parameters deduced before
    """

    ini_strict_sections_ = ["borders-reinterpolate"]

    #: Relative path to the input data files
    input_folder: Annotated[RelPath, Param.store(RelPath.param_type)]

    #: Indices of spectrum to process
    inputs: Annotated[
        Sequence[int],
        Param.append(
            types.int_.as_sequence_of_one(),
            positional=Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    #: Name of the output directory. Can be the same as the input folder, in which case files
    #: will be overwritten
    output_folder: Annotated[
        RelPath,
        Param.store(RelPath.param_type, short_flag_name="-o"),
    ]

    def validate_output_folder_(self) -> Optional[Err]:
        if self.root.at(self.output_folder).is_dir():
            return None  # all good
        else:
            return Err.make("The output directory needs to exist")

    #: Parameter file written by the borders_scan step
    parameter_file: Annotated[RelPath, Param.store(RelPath.param_type)]

    def validate_parameter_file_(self) -> Optional[Err]:
        if self.root.at(self.parameter_file).is_file():
            return None  # all good
        else:
            return Err.make("The parameter file needs to exist")


def cli() -> None:
    t = Task.from_command_line_()
    mt = Table.read_csv(t.root.at(t.input_master_table), BasicInfo)
    inputs: Sequence[int] = t.inputs
    if not inputs:
        inputs = list(range(mt.nrows()))
    for i in inputs:
        r = mt[i]
        if t.instrument in ["ESPRESSO", "EXPRESS"]:
            preprocess_fits_espresso_express(t, r)
        elif t.instrument in ["HARPS", "CORALIE", "HARPN"]:
            preprocess_fits_harps_coraline_harpn(t, r)
        else:
            raise ValueError(f"Instrument {t.instrument} not implemented")
