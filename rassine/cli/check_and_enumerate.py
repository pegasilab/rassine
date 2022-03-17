import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from re import I
from typing import Sequence, Set

import numpy as np
import pandas as pd
from configpile import AutoName, Config, Param, types
from typing_extensions import Annotated

from ..tybles import Table
from .base import BasicInfo, RassineConfigBeforeStack, RelPath


@dataclass(frozen=True)
class Task(RassineConfigBeforeStack):
    """
    Master table filename extraction tool

    This tool takes the path to a CSV file as an argument and returns the contents of the
    filename column, which are filenames without accompanying path, one per line.
    """

    ini_strict_sections_ = ["list-filenames"]

    #: Relative path to the folder containing the raw spectra
    input_folder: Annotated[RelPath, Param.store(RelPath.param_type, short_flag_name="-i")]

    #: Pattern for input files, by default look for FITS files
    glob_pattern: Annotated[str, Param.store(types.word, default_value="*.fits")]

    #: If true, we error if a file is present on the disk but not in the master table
    strict: Annotated[bool, Param.store(types.bool_, default_value="True")]


def run(t: Task) -> None:
    fn = t.root << t.input_master_table
    table = Table.read_csv(fn, BasicInfo)
    tb_files: Set[str] = {r.filename for r in table.all()}
    fs_files: Set[str] = {f.name for f in (t.root << t.input_folder).glob(t.glob_pattern)}
    if (t.strict and tb_files != fs_files) or (tb_files - fs_files):
        print("Error: mismatch between filenames in master table and on filesystem")
        for f in tb_files - fs_files:
            print(f"{f} is in the table but not in {t.root << t.input_folder}")
        for f in fs_files - tb_files:
            print(f"{f} is on the filesystem but not in the master table")
        sys.exit(1)
    for i in range(table.nrows()):
        print(i)


def cli() -> None:
    run(Task.from_command_line_())
