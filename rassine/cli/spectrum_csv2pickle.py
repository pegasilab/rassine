import pickle
from dataclasses import dataclass
from pathlib import Path

import configpile as cp
import numpy as np
import pandas as pd
from typing_extensions import Annotated


@dataclass(frozen=True)
class SpectrumCSVToPickle(cp.Config):
    """
    Converts a spectrum given as a CSV file to a pickle file that can be loaded by rassine_main

    The CSV column names are flexible, but the pickle dictionary keys will be "wave" and "flux".
    """

    #: Input CSV file to process
    input_file: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.validated(lambda p: p.exists(), "Input file must exist"),
            positional=cp.Positional.ONCE,
            long_flag_name=None,
        ),
    ]

    #: Output pickle to write
    output_file: Annotated[
        Path,
        cp.Param.store(cp.parsers.path_parser, positional=cp.Positional.ONCE, long_flag_name=None),
    ]

    #: Column name containing the flux
    flux_column_name: Annotated[
        str,
        cp.Param.store(cp.parsers.stripped_str_parser, short_flag_name="-f", default_value="flux"),
    ]

    #: Column name containing the wave
    wave_column_name: Annotated[
        str,
        cp.Param.store(cp.parsers.stripped_str_parser, short_flag_name="-w", default_value="wave"),
    ]

    #: Pickle protocol
    protocol: Annotated[
        int,
        cp.Param.store(
            cp.parsers.int_parser.validated(lambda i: i >= 0, "Must be valid pickle protocol"),
            default_value="5",
        ),
    ]


def cli() -> None:
    config = SpectrumCSVToPickle.from_command_line_()
    csv_data = pd.read_csv(config.input_file)
    spectrei = np.array(csv_data[config.flux_column_name])  # the flux of your spectrum
    grid = np.array(csv_data[config.wave_column_name])  # the grid of wavelength of your spectrum
    pickle_data = {"flux": spectrei, "wave": grid}
    with open(config.output_file, "wb") as f:
        pickle.dump(pickle_data, f, protocol=config.protocol)
