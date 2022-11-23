from __future__ import annotations

import argparse
import logging
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

import configpile as cp
import numpy as np
import tybles as tb
from typing_extensions import Annotated

from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.util import log_task_name_and_time
from .data import DACE, IndividualBasicRow

# CHECKME: how to sort spectra?


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Creates the individual spectrum table from the DACE table

    We assume that all raw files are in the same directory. We take the filenames from the DACE
    table but ignore the path information in the same DACE table.

    We also read the "model" column from the DACE table and compute rv_mean
    (actually the median) and rv_shift.
    """

    #
    # Common information
    #

    prog_ = Path(__file__).stem
    ini_strict_sections_ = [Path(__file__).stem]
    ini_relaxed_sections_ = [Path(__file__).stem.split("_")[0]]
    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], cp.Param.config(env_var_name="RASSINE_CONFIG")]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, cp.Param.root_path(env_var_name="RASSINE_ROOT")]

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

    #: Whether to verify that the filenames are sorted in the input DACE table
    #:
    #: This is an assumption that is made at several points in RASSINE/YARARA, and it is best
    #: to know whether this assumption is violated.
    verify_sorted: Annotated[bool, cp.Param.store(cp.parsers.bool_parser, default_value="true")]

    #: If true, we error if a file is present on the disk but not in the master table
    strict: Annotated[bool, cp.Param.store(cp.parsers.bool_parser, default_value="true")]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()

    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)

    dace_path = t.root / t.dace_table

    logging.info(f"Reading DACE table {dace_path}")
    dace = DACE.schema().read_csv(dace_path)
    if t.verify_sorted:
        filenames: List[str] = dace["fileroot"].to_list()
        if filenames != sorted(filenames):
            logging.error("The DACE table is not ordered by filenames")
            logging.error("(ote that the names contain the timestamps)")
            sys.exit(1)

    rows = tb.tyble(dace, DACE.schema())
    logging.info(f"{len(dace)} rows found")
    # verify that all files listed in the table are present
    tb_files: Set[str] = {Path(r.fileroot).name for r in rows}
    fs_files: Set[str] = {f.name for f in (t.root / t.input_folder).glob(t.glob_pattern)}
    if (t.strict and tb_files != fs_files) or (tb_files - fs_files):
        logging.error("Mismatch between filenames in master table and on filesystem")
        for f in tb_files - fs_files:
            logging.error(f"{f} is in the table but not in {t.root / t.input_folder}")
        for f in fs_files - tb_files:
            logging.error(f"{f} is on the filesystem but not in the master table")
        sys.exit(1)

    # compute rv_mean
    rv_mean: np.float64 = np.median(dace["model"])

    # create the rows in the individual table
    individual_rows = [
        IndividualBasicRow(
            name=Path(r.fileroot).stem,
            raw_filename=Path(r.fileroot).name,
            mjd=r.mjd,
            model=r.model,
            rv_mean=rv_mean,
            rv_shift=r.model - rv_mean,
            vrad=r.vrad,
            svrad=r.svrad,
            drift=r.drift_used,
        )
        for r in rows
    ]
    IndividualBasicRow.schema().from_rows(individual_rows, "DataFrame").to_csv(
        t.root / t.output_table, index=False
    )


def cli() -> None:
    run(Task.from_command_line_())
