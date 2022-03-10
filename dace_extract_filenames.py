import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from rassine.config import ParametersParser

logging.getLogger().setLevel(5)


def get_parser() -> ParametersParser:
    """
    Returns the argument parser used in this script
    """
    res = ParametersParser(
        strict_config_sections=["dace-extract-filenames"],
        description="""\
    DACE table filename extraction tool
    
    This tool takes the path to a DACE CSV file as an argument and returns the contents of the
    fileroot column, which are paths, one per line.
    
    The paths will be rewritten for the current system using the ``--root-path`` and ``--input-data-path`` options.
    """,
    )
    res.add_config_argument(
        "--config-file",
        "-c",
        type=Path,
        action="append",
        env_var_name="RASSINE_CONFIG_FILES",
        help="Use the specified configuration files. Files can be separated by commas/the command can be invoked multiple times.",
    )
    res.add_argument("--dace-input-file", type=Path, help="DACE CSV input file path")
    res.add_argument(
        "--root-path", env_var_name="RASSINE_ROOT_PATH", type=Path, help="Root path of the data"
    )
    res.add_argument("--input-data-path", type=Path, help="Relative path to the input data files")
    return res


if __name__ == "__main__":
    myparser = get_parser()
    myargs = myparser.parse_all()
    print(myargs)
    inputfile = myargs.root_path / myargs.dace_input_file
    newfolder = myargs.root_path / myargs.input_data_path
    files = np.sort(pd.read_csv(inputfile)["fileroot"])
    for f in files:
        print(str((newfolder / Path(f).name).resolve()))
