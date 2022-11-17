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
    Orders the rows of a CSV file according to the order given by a reference file

    Both files must have the same number of rows, and both must have a column of a given
    name with the same values, possibly ordered differently.
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

    #: Column to order by
    column: Annotated[str, cp.Param.store(cp.parsers.stripped_str_parser, short_flag_name="-c")]

    #: Reference file for ordering
    reference: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-r")]

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
    col = t.column
    ref = pd.read_csv(t.root / t.reference)
    file = pd.read_csv(t.root / t.file)
    assert col in ref.columns
    assert col in file.columns
    ref_values = ref[col]
    file_values = file[col]
    file_pos = {name: index for index, name in enumerate(file_values)}

    # reread file lines
    with open(t.root / t.file, "r") as f:
        lines = f.readlines()

    header = lines[0]
    data: Sequence[str] = lines[1:]
    output_lines: List[str] = [header]
    for value in list(ref_values):
        assert value in file_pos, f"Value {value} not in CSV file to reorder"
        output_lines.append(data[file_pos[value]])
        del file_pos[value]
    assert not file_pos, f"Values {file_pos.keys()} are present in the reference file"

    with open(t.root / t.file, "w") as f:
        f.writelines(output_lines)


def cli() -> None:
    run(Task.from_command_line_())
