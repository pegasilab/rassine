import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from configpile import types
from configpile.arg import AutoName, Param
from configpile.config import Config
from typing_extensions import Annotated

from rassine.config import ParametersParser

logging.getLogger().setLevel(5)


@dataclass(frozen=True)
class Dace(Config):
    """
    DACE table filename extraction tool

    This tool takes the path to a DACE CSV file as an argument and returns the contents of the
    fileroot column, which are paths, one per line.

    The paths will be rewritten for the current system using the ``--root-path`` and ``--input-data-path`` options.
    """

    env_prefix_ = "RASSINE"
    ini_strict_sections_ = ["dace-extract-filenames"]

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], Param.config(env_var_name=AutoName.DERIVED)]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, Param.store(types.path, env_var_name=AutoName.DERIVED)]

    #: DACE CSV input file path
    dace_input_file: Annotated[Path, Param.store(types.path, env_var_name=AutoName.DERIVED)]

    #: Relative path to the input data files
    input_folder: Annotated[Path, Param.store(types.path)]


if __name__ == "__main__":
    config = Dace.from_command_line_()
    inputfile = config.root / config.dace_input_file
    newfolder = config.root / config.input_folder
    files = np.sort(pd.read_csv(inputfile)["fileroot"])
    for f in files:
        print(str((newfolder / Path(f).name).resolve()))
