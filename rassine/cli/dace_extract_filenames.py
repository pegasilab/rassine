import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from configpile import AutoName, Config, Param, types
from typing_extensions import Annotated

from .base import RassineConfig, RelPath


@dataclass(frozen=True)
class Task(RassineConfig):
    """
    DACE table filename extraction tool

    This tool takes the path to a DACE CSV file as an argument and returns the contents of the
    fileroot column, which are paths, one per line.

    The paths will be rewritten for the current system using the ``--root-path`` and
    ``--input-folder`` options.
    """

    ini_strict_sections_ = ["dace-extract-filenames"]

    #: DACE CSV input file path
    dace_input_file: Annotated[
        RelPath, Param.store(RelPath.param_type, env_var_name=AutoName.DERIVED)
    ]

    def run(self) -> None:
        inputfile = self.root << self.dace_input_file
        files = np.sort(pd.read_csv(inputfile)["fileroot"])
        for f in files:
            print(Path(f).name)

    @staticmethod
    def cli() -> None:
        Task.from_command_line_().run()
