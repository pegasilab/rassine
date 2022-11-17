from __future__ import annotations

import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import configpile as cp
import numpy as np
import tybles as tb
from typing_extensions import Annotated

from ..imports.reinterpolate import IndividualReinterpolatedRow
from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.math import Float
from ..lib.util import log_task_name_and_time
from .data import IndividualGroupRow


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Stacking: preparation of the groups
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

    #: Reinterpolated table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Output grouping table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Length of the binning for the stacking in days
    #:
    #: Can be a floating point value indicating fractions of a day
    #:
    #: The value 0 means that each spectrum is in its individual group
    bin_length: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="1.0")]

    #: dbin to shift the binning (0.5 for solar data)
    dbin: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="0.0")]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()

    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)

    input_table_path = t.root / t.input_table

    logging.info(f"Reading table {input_table_path}")
    rows = IndividualReinterpolatedRow.schema().read_csv(input_table_path, return_type="Tyble")

    if t.bin_length == 0:
        group_info = [
            IndividualGroupRow(r.name, i, np.float64(t.bin_length), np.float64(t.dbin))
            for i, r in enumerate(rows)
        ]
    else:

        def compute_group(jdb: Float) -> int:
            return int((jdb + t.dbin) // t.bin_length)

        group_info = [
            IndividualGroupRow(
                r.name, compute_group(r.jdb), np.float64(t.bin_length), np.float64(t.dbin)
            )
            for r in rows
        ]

    IndividualGroupRow.schema().from_rows(group_info, "DataFrame").to_csv(
        t.root / t.output_table, index=False
    )


def cli() -> None:
    run(Task.from_command_line_())
