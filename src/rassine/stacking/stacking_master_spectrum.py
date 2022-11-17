from __future__ import annotations

import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import tybles as tb
from typing_extensions import Annotated

from ..lib.analysis import find_nearest1
from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import create_grid
from ..lib.util import log_task_name_and_time
from .data import MasterPickle, MasterRow, StackedBasicRow, StackedPickle


@dataclass(frozen=True)
class Task(cp.Config):
    """Creates a master spectrum from stacked spectra"""

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

    #: Pickle protocol version to use
    pickle_protocol: Annotated[
        PickleProtocol, cp.Param.store(PickleProtocol.parser(), default_value="3")
    ]

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

    #: Input stacked basic table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Output master spectrum info table (one row)
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Folder containing the stacked spectra
    input_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]

    #: Path to the master spectrum file to write
    output_file: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-o")]

    #: Group indices to process
    #:
    #: If not provided, all groups are processed
    groups: Annotated[
        Sequence[int],
        cp.Param.append1(
            cp.parsers.int_parser,
            positional=cp.Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()
    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)
    (t.root / t.output_file).parent.mkdir(parents=True, exist_ok=True)

    logging.debug(f"Reading {t.root/t.input_table}")
    input_tyble = StackedBasicRow.schema().read_csv(t.root / t.input_table, return_type="Tyble")
    first = input_tyble[0]
    nb_bins = first.nb_bins
    stack = np.zeros(nb_bins, dtype=np.float64)
    berv_mins = [r.berv_min for r in input_tyble]
    berv_maxs = [r.berv_max for r in input_tyble]
    nb_spectra = [r.nb_spectra_stacked for r in input_tyble]
    all_berv = np.array([r.berv for r in input_tyble])
    all_snr = np.array([r.SNR_5500 for r in input_tyble])
    wave_min = input_tyble[0].wave_min
    dwave = input_tyble[0].dwave
    nb_bins = input_tyble[0].nb_bins
    grid = create_grid(wave_min, dwave, int(nb_bins))

    for row in input_tyble:
        data = open_pickle(t.root / t.input_folder / f"{row.name}.p", StackedPickle)
        stack += data["flux"]

    stack[stack <= 0.0] = 0.0
    wave_ref = int(find_nearest1(grid, 5500)[0])
    continuum_5500 = np.nanpercentile(stack[wave_ref - 50 : wave_ref + 50], 95)
    SNR = np.sqrt(continuum_5500)
    BERV = np.sum(all_berv * all_snr**2) / np.sum(all_snr**2)
    BERV_MIN = np.min(berv_mins)
    BERV_MAX = np.max(berv_maxs)
    out: MasterPickle = {
        "flux": stack,
        "master_spectrum": True,
        "RV_sys": first.rv_mean,
        "RV_shift": np.float64(0.0),
        "SNR_5500": SNR,
        "lamp_offset": np.float64(0.0),
        "acc_sec": first.acc_sec,
        "berv": BERV,
        "berv_min": BERV_MIN,
        "berv_max": BERV_MAX,
        "instrument": first.instrument,
        "mjd": np.float64(0.0),
        "jdb": np.float64(0.0),
        "hole_left": first.hole_left,
        "hole_right": first.hole_right,
        "wave_min": first.wave_min,
        "wave_max": first.wave_max,
        "dwave": first.dwave,
        "nb_spectra_stacked": int(np.sum(nb_spectra)),
        "arcfiles": "none",
    }
    save_pickle(t.root / t.output_file, out)
    out_row = MasterRow(
        name=t.output_file.stem,
        SNR_5500=SNR,
        acc_sec=first.acc_sec,
        berv=BERV,
        berv_min=BERV_MIN,
        berv_max=BERV_MAX,
        instrument=first.instrument,
        hole_left=first.hole_left,
        hole_right=first.hole_right,
        wave_min=first.wave_min,
        wave_max=first.wave_max,
        dwave=first.dwave,
        nb_spectra_stacked=int(np.sum(nb_spectra)),
    )
    MasterRow.schema().from_rows([out_row]).to_csv(t.root / t.output_table, index=False)


def cli() -> None:
    run(Task.from_command_line_())
