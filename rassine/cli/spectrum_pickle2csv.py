import pickle
from dataclasses import dataclass
from pathlib import Path

import configpile as cp
import configpile.parsers as cpp
import numpy as np
import pandas as pd
from typing_extensions import Annotated


@dataclass(frozen=True)
class SpectrumPickleToCSV(cp.Config):
    """
    Converts a spectrum given as a pickle file to a CSV file

    The CSV column names are flexible, but the pickle dictionary keys will be "wave" and "flux".
    """

    #: Input pickle file to process
    input_file: Annotated[
        Path,
        cp.Param.store(
            cpp.path_parser.validated(lambda p: p.exists(), "Input file must exist"),
            positional=cp.Positional.ONCE,
            long_flag_name=None,
        ),
    ]

    #: Output CSV to write
    output_file: Annotated[
        Path, cp.Param.store(cpp.path_parser, positional=cp.Positional.ONCE, long_flag_name=None)
    ]

    #: Column name containing the flux
    flux_column_name: Annotated[
        str, cp.Param.store(cpp.stripped_str_parser, short_flag_name="-f", default_value="flux")
    ]

    #: Column name containing the wave
    wave_column_name: Annotated[
        str, cp.Param.store(cpp.stripped_str_parser, short_flag_name="-w", default_value="wave")
    ]

    #: Pickle protocol
    protocol: Annotated[
        int,
        cp.Param.store(
            cpp.int_parser.validated(lambda i: i >= 0, "Must be valid pickle protocol"),
            default_value="5",
        ),
    ]


def cli() -> None:
    config = SpectrumPickleToCSV.from_command_line_()
    with open(config.input_file, "rb") as f:
        pickle_data = pickle.load(f)
    csv_data = pd.DataFrame(
        {
            config.flux_column_name: pickle_data["flux"],
            config.wave_column_name: pickle_data["wave"],
        }
    )
    csv_data.to_csv(config.output_file)
