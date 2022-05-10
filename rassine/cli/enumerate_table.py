import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import configpile as cp
import pandas as pd
from typing_extensions import Annotated

from rassine.cli.util import log_task_name_and_time

from .data import LoggingLevel


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Enumerates the row indices of a CSV table, one per row, into the standard output

    Specifically, this prints the integers 0 to n-1 included, where n is the number of rows
    in the CSV file.
    """

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

    prog_ = "enumerate_table"

    #: Input table to enumerate the row indices of
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]


@log_task_name_and_time(name="enumerate_table")
def run(t: Task) -> None:
    t.logging_level.set()
    table_path = t.root / t.input_table
    table = pd.read_csv(table_path)
    logging.info(f"Enumerating rows of table {table_path} with {len(table)} rows")
    for i in range(0, len(table)):
        print(i)


def cli() -> None:
    run(Task.from_command_line_())
