from __future__ import annotations

import argparse
import logging
import typing  # pylint: disable=W0611
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import tybles as tb
from astropy.time import Time
from filelock import FileLock
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Annotated

from ..imports.reinterpolate import IndividualReinterpolatedRow, ReinterpolatedSpectrumPickle
from ..lib.analysis import find_nearest1
from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import create_grid
from ..lib.util import log_task_name_and_time
from .data import IndividualGroupRow, StackedBasicRow, StackedPickle


@dataclass(frozen=True)
class Task(cp.Config):
    """Stacks spectra"""

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

    #: Input reinterpolated table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Group description table
    group_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-G")]

    #: Output stacked table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Relative path to the folder containing the reinterpolated spectra
    input_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-i")]

    #: Name of the output directory
    output_folder: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-o")]

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


def perform_stacking(
    t: Task,
    rows: Sequence[IndividualReinterpolatedRow],
    group: int,
    bin_length: np.float64,
    dbin: np.float64,
) -> StackedBasicRow:

    nb_spectra_stacked = len(rows)

    def input_path(row: IndividualReinterpolatedRow) -> Path:
        return t.root / t.input_folder / (row.name + ".p")

    file = input_path(rows[0])
    data = open_pickle(file, ReinterpolatedSpectrumPickle)
    nb_bins = rows[0].nb_bins
    flux = data["flux"]
    wave_min = data["wave_min"]
    wave_max = data["wave_max"]
    dwave = data["dwave"]
    grid = create_grid(wave_min, dwave, len(flux))
    RV_sys = data["RV_sys"]
    instrument = data["instrument"]
    hole_left = data["hole_left"]
    hole_right = data["hole_right"]
    acc_sec = data["acc_sec"]

    stack = flux
    stack_err2 = data["flux_err"] ** 2

    def compute_bolo(f: NDArray[np.float64]) -> np.float64:
        return np.nansum(f) / len(f)

    name_root_files = [str(file)]
    bolo_ = [compute_bolo(flux)]
    for row in rows[1:]:
        file = input_path(row)
        data = open_pickle(file, ReinterpolatedSpectrumPickle)
        flux = data["flux"]
        stack += flux
        stack_err2 += data["flux_err"] ** 2
        bolo_.append(compute_bolo(flux))
        name_root_files.append(str(file))

    bolo = np.array(bolo_)

    def bolometric_weighted_average(values: ArrayLike) -> np.float64:
        """Computes the bolometric average of a value"""
        return np.sum(np.asarray(values) * bolo) / np.sum(bolo)

    jdb_w = bolometric_weighted_average([r.jdb for r in rows])
    date_name = Time(jdb_w - 0.5, format="mjd").isot
    berv = [r.berv for r in rows]
    berv_w = bolometric_weighted_average(berv)
    lamp_w = bolometric_weighted_average([r.lamp_offset for r in rows])
    rv_shift_w = bolometric_weighted_average([r.rv_shift for r in rows])
    wave_ref = int(find_nearest1(grid, 5500)[0])
    continuum_5500 = np.nanpercentile(stack[wave_ref - 50 : wave_ref + 50], 95)
    SNR = np.sqrt(continuum_5500)
    mjd_w = jdb_w - 0.5

    jdb_array = np.array([r.jdb for r in rows], dtype=np.float64)
    vrad_array = np.array([r.vrad for r in rows], dtype=np.float64)
    svrad_array = np.array([r.svrad for r in rows], dtype=np.float64)
    model_array = np.array([r.model for r in rows], dtype=np.float64)
    drift_array = np.array([r.drift for r in rows], dtype=np.float64)

    vrad_array += drift_array - model_array * 1000.0
    weights_array = 1.0 / svrad_array**2

    def uncertainty_weighted_average(values: NDArray[np.float64]) -> np.float64:
        """Computes the weighted average according to the radial vel. uncertainties"""
        return np.sum(values * weights_array) / np.sum(weights_array)

    mean_jdb = uncertainty_weighted_average(jdb_array)
    mean_vrad = uncertainty_weighted_average(vrad_array)
    mean_svrad = 1 / np.sqrt(np.sum(weights_array))

    # Computations taken from YARARA

    out: StackedPickle = {
        "flux": stack,
        "flux_err": np.sqrt(np.abs(stack_err2)),
        "jdb": jdb_w,
        "mjd": mjd_w,
        "berv": berv_w,
        "lamp_offset": lamp_w,
        "acc_sec": acc_sec,
        "RV_shift": rv_shift_w,
        "RV_sys": RV_sys,
        "SNR_5500": SNR,
        "hole_left": hole_left,
        "hole_right": hole_right,
        "wave_min": wave_min,
        "wave_max": wave_max,
        "dwave": dwave,
        "stacking_length": bin_length,
        "nb_spectra_stacked": int(nb_spectra_stacked),
        "arcfiles": name_root_files,
    }
    name = f"Stacked_spectrum_B{bin_length:.2f}_D{date_name}"
    output_file = t.root / t.output_folder / f"{name}.p"
    save_pickle(output_file, out)

    return StackedBasicRow(
        name=name,
        group=group,
        instrument=instrument,
        rv_mean=RV_sys,
        rv_shift=rv_shift_w,
        SNR_5500=SNR,
        berv=berv_w,
        berv_min=np.min(berv),
        berv_max=np.max(berv),
        lamp_offset=lamp_w,
        acc_sec=acc_sec,
        mjd=mjd_w,
        jdb=jdb_w,
        hole_left=hole_left,
        hole_right=hole_right,
        wave_min=wave_min,
        wave_max=wave_max,
        dwave=dwave,
        stacking_length=bin_length,
        nb_spectra_stacked=np.int64(nb_spectra_stacked),
        nb_bins=nb_bins,
        mean_jdb=mean_jdb,
        mean_vrad=mean_vrad,
        mean_svrad=mean_svrad,
    )


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()
    (t.root / t.output_folder).mkdir(parents=True, exist_ok=True)
    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)

    logging.debug("Reading %s", t.root / t.input_table)
    input_tyble = IndividualReinterpolatedRow.schema().read_csv(
        t.root / t.input_table, return_type="Tyble"
    )
    input_df = input_tyble.data_frame

    logging.debug("Reading %s", t.root / t.group_table)
    group_tyble = IndividualGroupRow.schema().read_csv(t.root / t.group_table, return_type="Tyble")
    group_df = group_tyble.data_frame

    bin_length = group_df["stacking_length"][0]
    assert np.all(group_df["stacking_length"] == bin_length), "stacking_length must be uniform"

    dbin = group_df["dbin"][0]
    assert np.all(group_df["dbin"] == dbin), "dbin must be uniform"

    assert input_df["name"].equals(
        group_df["name"]
    ), "The input and the group tables must be consistent"

    if not t.groups:
        groups: Sequence[int] = list(group_df["group"].unique())
    else:
        groups = t.groups

    stacked_rows: List[StackedBasicRow] = []
    for group in groups:
        rows = [input_tyble[i] for i in group_df.index[group_df["group"] == group]]
        assert rows, "A group must have at least one spectrum in it"
        stacked_rows.append(perform_stacking(t, rows, group, bin_length, dbin))

    output_table = t.root / t.output_table
    output_table_lockfile = output_table.with_suffix(output_table.suffix + ".lock")

    logging.debug("Appending to output table %s", output_table)
    with FileLock(output_table_lockfile):
        df = StackedBasicRow.schema().from_rows(stacked_rows, return_type="DataFrame")
        df.to_csv(output_table, header=not output_table.exists(), mode="a", index=False)


def cli() -> None:
    run(Task.from_command_line_())
