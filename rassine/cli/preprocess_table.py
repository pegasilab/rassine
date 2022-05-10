from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Set

import configpile as cp
import numpy as np
import tybles as tb
from typing_extensions import Annotated

from .data import LoggingLevel, PickleProtocol
from .util import log_task_name_and_time

# CHECKME: how to sort spectra?


@dataclass(frozen=True)
class DACE:
    """Columns of the DACE table we import"""

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction
    model: np.float64

    @staticmethod
    def schema() -> tb.Schema[DACE]:
        return tb.schema(DACE, order_columns=True, missing_columns="error", extra_columns="drop")


@dataclass(frozen=True)
class Individual:
    """Columns of the DACE data we extract for the rest of the processing"""

    #: Spectrum name without path and extension
    name: str

    #: Raw filename
    raw_filename: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction
    model: np.float64

    @staticmethod
    def schema() -> tb.Schema[Individual]:
        return tb.schema(
            Individual, order_columns=True, missing_columns="error", extra_columns="drop"
        )


@dataclass(frozen=True)
class Task(cp.Config):
    """Creates the individual spectrum table from the DACE table"""

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

    prog_ = "preprocess_table"

    ini_strict_sections_ = ["preprocess"]

    #: Input DACE table
    dace_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Relative path to the folder containing the raw spectra
    input_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]

    #: Output individual table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Pattern for input files, by default look for FITS files
    glob_pattern: Annotated[
        str, cp.Param.store(cp.parsers.stripped_str_parser, default_value="*.fits")
    ]

    #: If true, we error if a file is present on the disk but not in the master table
    strict: Annotated[bool, cp.Param.store(cp.parsers.bool_parser, default_value="true")]


@log_task_name_and_time(name="preprocess_import")
def run(t: Task) -> None:
    t.logging_level.set()
    dace_path = t.root / t.dace_table
    logging.info(f"Reaading DACE table {dace_path}")
    dace = DACE.schema().read_csv(dace_path)
    rows = tb.tyble(dace, DACE.schema())
    logging.info(f"{len(dace)} rows found")
    tb_files: Set[str] = {Path(r.fileroot).name for r in rows}
    fs_files: Set[str] = {f.name for f in (t.root / t.input_folder).glob(t.glob_pattern)}
    if (t.strict and tb_files != fs_files) or (tb_files - fs_files):
        print("Error: mismatch between filenames in master table and on filesystem")
        for f in tb_files - fs_files:
            print(f"{f} is in the table but not in {t.root / t.input_folder}")
        for f in fs_files - tb_files:
            print(f"{f} is on the filesystem but not in the master table")
        sys.exit(1)

    individual_rows = [
        Individual(
            name=Path(r.fileroot).stem,
            raw_filename=Path(r.fileroot).name,
            mjd=r.mjd,
            model=r.model,
        )
        for r in rows
    ]
    Individual.schema().from_rows(individual_rows, "DataFrame").to_csv(t.root / t.output_table)


def cli() -> None:
    run(Task.from_command_line_())
