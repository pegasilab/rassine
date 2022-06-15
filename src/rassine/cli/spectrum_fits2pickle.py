import pickle
from dataclasses import dataclass
from pathlib import Path

import configpile as cp
import numpy as np
from astropy.io import fits
from typing_extensions import Annotated


@dataclass(frozen=True)
class SpectrumFITSToPickle(cp.Config):
    """
    Converts a spectrum given as a FITS file to a pickle file that can be loaded by rassine_main
    """

    #: Input FITS file to process
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

    #: Pickle protocol
    protocol: Annotated[
        int,
        cp.Param.store(
            cp.parsers.int_parser.validated(lambda i: i >= 0, "Must be valid pickle protocol"),
            default_value="5",
        ),
    ]


def cli() -> None:
    config = SpectrumFITSToPickle.from_command_line_()
    header = fits.getheader(config.input_file)
    data = fits.getdata(config.input_file)
    spectre_step = header["CDELT1"]
    spectrei = data.astype("float64")  # the flux of your spectrum
    grid = np.linspace(
        header["CRVAL1"], header["CRVAL1"] + (len(spectrei) - 1) * spectre_step, len(spectrei)
    )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)
    pickle_data = {"flux": spectrei, "wave": grid}
    with open(config.output_file, "wb") as f:
        pickle.dump(pickle_data, f, protocol=config.protocol)
