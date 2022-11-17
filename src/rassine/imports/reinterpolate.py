from __future__ import annotations

import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import tybles as tb
from filelock import FileLock
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing_extensions import Annotated

from ..lib.analysis import find_nearest1
from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import absurd_minus_99_9, create_grid, doppler_r
from ..lib.util import log_task_name_and_time
from .data import (
    IndividualImportedRow,
    IndividualReinterpolatedRow,
    PickledIndividualSpectrum,
    ReinterpolatedSpectrumPickle,
)


@dataclass(frozen=True)
class Task(cp.Config):
    """Reinterpolates the spectra to match the stellar frame"""

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

    #: Input spectrum table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Output spectrum table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Relative path to the folder containing the spectra
    input_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]

    #: Name of the output directory. If None, the output directory is created at the same location than the spectra.
    output_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-o")]

    #: Indices of spectrum to process
    #:
    #: If not provided, all spectra are processed
    inputs: Annotated[
        Sequence[int],
        cp.Param.append1(
            cp.parsers.int_parser,
            positional=cp.Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    #: Wavelength step in angstrom used to produce the equidistant wavelength vector
    dlambda: Annotated[float, cp.Param.store(cp.parsers.float_parser)]

    def validate_output_folder(self) -> Optional[cp.Err]:
        return cp.Err.check(
            (self.root / self.output_folder).is_dir(), "The output directory needs to exist"
        )


@dataclass(frozen=True)
class ReinterpolationSettings:
    wave_min_k: np.float64
    wave_max_k: np.float64
    hole_left_k: np.float64
    hole_right_k: np.float64
    dlambda: np.float64
    nb_bins: int
    static_grid: NDArray[np.float64]
    wave_ref: int

    @staticmethod
    def make(
        wave_min_k: np.float64,
        wave_max_k: np.float64,
        hole_left_k: np.float64,
        hole_right_k: np.float64,
        dlambda: np.float64,
        nb_bins: int,
    ) -> ReinterpolationSettings:
        static_grid = create_grid(wave_min_k, dlambda, nb_bins)
        wave_ref = find_nearest1(static_grid, 5500)[0]
        return ReinterpolationSettings(
            wave_min_k=wave_min_k,
            wave_max_k=wave_max_k,
            hole_left_k=hole_left_k,
            hole_right_k=hole_right_k,
            dlambda=dlambda,
            nb_bins=nb_bins,
            static_grid=static_grid,
            wave_ref=wave_ref,
        )

    @staticmethod
    def from_spectrum_data(
        wave_min: NDArray[np.float64],
        wave_max: NDArray[np.float64],
        hole_left: NDArray[np.float64],
        hole_right: NDArray[np.float64],
        dlambda: np.float64,
    ) -> ReinterpolationSettings:
        hole_left = hole_left[hole_left != absurd_minus_99_9]
        hole_right = hole_right[hole_right != absurd_minus_99_9]

        if len(hole_left) != 0 and len(hole_right) != 0:
            hole_left_k = np.min(hole_left) - 0.5  # increase of 0.5 the gap limit by security
            hole_right_k = np.max(hole_right) + 0.5  # increase of 0.5 the gap limit by security
        else:
            hole_left_k = absurd_minus_99_9
            hole_right_k = absurd_minus_99_9
        wave_min_k = wave_min.max()
        wave_max_k = wave_max.min()
        nb_bins: int = int(np.ceil((wave_max_k - wave_min_k) / dlambda))
        return ReinterpolationSettings.make(
            wave_min_k=wave_min_k,
            wave_max_k=wave_max_k,
            hole_left_k=np.float64(hole_left_k),
            hole_right_k=np.float64(hole_right_k),
            dlambda=dlambda,
            nb_bins=nb_bins,
        )


def reinterpolate(
    t: Task, row: IndividualImportedRow, s: ReinterpolationSettings
) -> IndividualReinterpolatedRow:
    input_file = t.root / t.input_folder / (row.name + ".p")
    output_file = t.root / t.output_folder / (row.name + ".p")
    spectrum = open_pickle(input_file, PickledIndividualSpectrum)

    # raw spectrum
    wave = spectrum["wave"]
    flux = spectrum["flux"]
    flux_err = spectrum["flux_err"]

    # reinterpolated using static_grid
    static_grid = s.static_grid
    new_flux = interp1d(
        doppler_r(wave, row.rv_shift)[1],
        flux,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )(static_grid)
    new_flux_err = interp1d(
        doppler_r(wave, row.rv_shift)[1],
        flux_err,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(static_grid)

    mask2 = (static_grid >= (s.hole_left_k - s.dlambda / 2.0)) & (
        static_grid <= (s.hole_right_k + s.dlambda / 2.0)
    )
    new_flux[mask2] = 0
    new_flux_err[mask2] = 1

    continuum_5500 = np.nanpercentile(new_flux[s.wave_ref - 50 : s.wave_ref + 50], 95)
    SNR = np.sqrt(continuum_5500)

    out: ReinterpolatedSpectrumPickle = {
        "flux": new_flux,
        "flux_err": new_flux_err,
        "RV_sys": row.rv_mean,
        "RV_shift": row.rv_shift,
        "SNR_5500": SNR,
        "berv": row.berv,
        "lamp_offset": row.lamp_offset,
        "plx_mas": row.plx_mas,
        "acc_sec": row.acc_sec,
        "instrument": row.instrument,
        "mjd": row.mjd,
        "jdb": row.jdb,
        "hole_left": s.hole_left_k,
        "hole_right": s.hole_right_k,
        "wave_min": s.wave_min_k,
        "wave_max": s.wave_max_k,
        "dwave": s.dlambda,
    }
    save_pickle(output_file, out)
    return IndividualReinterpolatedRow(
        name=row.name,
        instrument=row.instrument,
        mjd=row.mjd,
        jdb=row.jdb,
        model=row.model,
        rv_mean=row.rv_mean,
        rv_shift=row.rv_shift,
        SNR_5500=SNR,
        berv=row.berv,
        lamp_offset=row.lamp_offset,
        plx_mas=row.plx_mas,
        acc_sec=row.acc_sec,
        wave_min=s.wave_min_k,
        wave_max=s.wave_max_k,
        dwave=s.dlambda,
        nb_bins=np.int64(s.nb_bins),
        hole_left=s.hole_left_k,
        hole_right=s.hole_right_k,
        vrad=row.vrad,
        svrad=row.svrad,
        drift=row.drift,
    )


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()
    (t.root / t.output_folder).mkdir(parents=True, exist_ok=True)
    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)

    logging.debug(f"Reading {t.root/t.input_table}")
    tyble = IndividualImportedRow.schema().read_csv(t.root / t.input_table, return_type="Tyble")
    df = tyble.data_frame

    # compute hole boundaries
    settings = ReinterpolationSettings.from_spectrum_data(
        wave_min=np.round(df["wave_min"].to_numpy(), 8).astype(
            np.float64
        ),  # to take into account float32
        wave_max=np.round(df["wave_max"].to_numpy(), 8).astype(
            np.float64
        ),  # to take into account float32
        hole_left=df["hole_left"].to_numpy(),
        hole_right=df["hole_right"].to_numpy(),
        dlambda=np.float64(t.dlambda),
    )

    inputs: Sequence[int] = t.inputs
    if not inputs:
        inputs = list(range(len(df)))

    output_rows: List[IndividualReinterpolatedRow] = []
    for i in inputs:
        r = tyble[i]
        output_rows.append(reinterpolate(t, r, settings))

    output_table = t.root / t.output_table
    output_table_lockfile = output_table.with_suffix(output_table.suffix + ".lock")

    logging.debug(f"Appending to output table {output_table}")
    with FileLock(output_table_lockfile):
        df = IndividualReinterpolatedRow.schema().from_rows(output_rows, return_type="DataFrame")
        df.to_csv(output_table, header=not output_table.exists(), mode="a", index=False)


def cli() -> None:
    run(Task.from_command_line_())
