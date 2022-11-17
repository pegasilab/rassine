import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import configpile as cp
import pandas as pd
from typing_extensions import Annotated

from ..lib.data import LoggingLevel
from ..lib.util import log_task_name_and_time


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

    #: Input table to enumerate the row indices of
    input_table: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser,
            long_flag_name=None,
            short_flag_name=None,
            positional=cp.Positional.ONCE,
        ),
    ]

    #: Column to get unique values from
    column: Annotated[str, cp.Param.store(cp.parsers.stripped_str_parser, short_flag_name="-c")]


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    table_path = t.root / t.input_table
    table = pd.read_csv(table_path)
    unique_values = table[t.column].unique()
    logging.debug(f"Got {len(unique_values)} unique values from {len(table)} rows of {table_path}")
    for v in unique_values:
        print(v)


def cli() -> None:
    run(Task.from_command_line_())
