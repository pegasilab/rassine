import argparse
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import configpile as cp
import pandas as pd
from typing_extensions import Annotated

from ..lib.data import LoggingLevel
from ..lib.util import log_task_name_and_time


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Sorts the rows of a CSV file according to a specified column
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

    #: Column to sort by
    column: Annotated[str, cp.Param.store(cp.parsers.stripped_str_parser, short_flag_name="-c")]

    #: CSV file to reorder
    file: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser,
            short_flag_name=None,
            long_flag_name=None,
            positional=cp.Positional.ONCE,
        ),
    ]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    data = pd.read_csv(t.root / t.file)
    assert t.column in data.columns
    data.sort_values(t.column, inplace=True)

    # reread file lines
    with open(t.root / t.file, "r") as f:
        lines = f.readlines()

    header = lines[0]
    output_lines: List[str] = [header, *[lines[i + 1] for i in data.index]]
    with open(t.root / t.file, "w") as f:
        f.writelines(output_lines)


def cli() -> None:
    run(Task.from_command_line_())
