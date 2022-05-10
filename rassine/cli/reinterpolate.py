from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
import tybles as tb
from astropy.io import fits
from astropy.time import Time
from filelock import FileLock
from numpy.typing import NDArray
from typing_extensions import Annotated

from rassine.math import create_grid

from ..analysis import grouping
from ..data import absurd_minus_99_9
from .data import LoggingLevel, PickleProtocol
from .preprocess_table import Individual
from .util import log_task_name_and_time


class PickledReinterpolatedSpectrum(TypedDict):
    """
    Data format of the pickle files produced by the reinterpolation step
    """


@dataclass(frozen=True)
class Task(cp.Config):
    """Reinterpolates the spectra to match the stellar frame"""

    #
    # Common information
    #

    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], cp.Param.config(env_var_name="RASSINE_CONFIG")]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, cp.Param.store(cp.parsers.path_parser, env_var_name="RASSINE_ROOT")]

    #: Pickle protocol version to use
    pickle_protocol: Annotated[
        PickleProtocol, cp.Param.store(PickleProtocol.parser(), default_value="3")
    ]

    #: Logging level to use
    logging_level: Annotated[
        LoggingLevel,
        cp.Param.store(
            LoggingLevel.parser(), default_value="WARNING", env_var_name="RASSINE_LOGGING_LEVEL"
        ),
    ]

    #
    # Task specific information
    #

    prog_ = "reinterpolate"

    ini_strict_sections_ = ["reinterpolate"]

    #: Input spectrum table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Output spectrum table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Relative path to the folder containing the raw spectra
    input_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]

    #: Name of the output directory. If None, the output directory is created at the same location than the spectra.
    output_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-o")]

    #: Indices of spectrum to process
    #:
    #: If not provided, all spectra are processed
    inputs: Annotated[
        Sequence[int],
        cp.Param.append1(
            cp.parsers.int_parser,
            positional=cp.Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    #: Instrument format of the s1d spectra
    instrument: Annotated[
        str, cp.Param.store(cp.parsers.stripped_str_parser, default_value="HARPS")
    ]

    #: Parallax in mas (no more necessary ?)
    plx_mas: Annotated[
        float,
        cp.Param.store(cp.parsers.float_parser, default_value="0.0"),
    ]

    def validate_output_folder(self) -> Optional[cp.Err]:
        return cp.Err.check(
            (self.root / self.output_folder).is_dir(), "The output directory needs to exist"
        )
