from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Sequence, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import tybles as tb
from astropy.time import Time
from filelock import FileLock
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Annotated

from ..analysis import find_nearest1
from ..io import open_pickle, save_pickle
from ..math import create_grid
from .data import LoggingLevel, PickleProtocol
from .reinterpolate import IndividualReinterpolatedRow, PickledReinterpolatedSpectrum
from .stacking_create_groups import IndividualGroupRow
from .stacking_stack import StackedBasicRow, StackedPickle
from .util import log_task_name_and_time


@dataclass(frozen=True)
class MasterRow:
    SNR_5500: np.float64
    acc_sec: np.float64
    berv: np.float64
    berv_min: np.float64
    berv_max: np.float64
    instrument: str
    hole_left: np.float64
    hole_right: np.float64
    wave_min: np.float64
    wave_max: np.float64
    dwave: np.float64
    nb_spectra_stacked: int

    @staticmethod
    def schema() -> tb.Schema[MasterRow]:
        return tb.schema(
            MasterRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


class MasterPickle(TypedDict):
    """
    Data format of the pickle files produced by the stacking step

    All the weighted averages are made using the bolometric flux
    """

    #: Flux, stacked
    flux: npt.NDArray[np.float64]
    #: Tells it is the master spectrum
    master_spectrum: Literal[True]
    #: Average rv correction (median), same for all spectra
    RV_sys: np.float64
    #: RV correction, shift compared to the median, weighted average
    RV_shift: Literal[0]
    #: Corresponds to the square root of the 95th percentile for 100 bins around the wavelength=5500
    SNR_5500: np.float64
    #: lamp_offset, weighted average
    lamp_offset: Literal[0]
    #: acc_sec, taken from first spectrum
    acc_sec: np.float64
    #: berv, weighted average according to SNR
    berv: np.float64
    #: np.min(berv) of the individual spectra
    berv_min: np.float64
    #: np.max(berv) of the individual spectra
    berv_max: np.float64
    #: Instrument
    instrument: str
    #: mjd weighted average
    mjd: Literal[0]
    #: jdb weighted average
    jdb: Literal[0]
    #: Left boundary of hole, or -99.9 if not present
    hole_left: np.float64
    #: Right boundary of hole, or -99.9 if not present
    hole_right: np.float64
    #: Minimum wavelength
    wave_min: np.float64
    #: Maximum wavelength, not necessarily equal to np.max(static_grid)
    wave_max: np.float64
    #: delta between two bins, synonym dlambda
    dwave: np.float64
    #: Number of individual spectra using for this individual spectrum
    nb_spectra_stacked: int
    #: Paths of files used in this stacked spectrum
    arcfiles: Literal["none"]


@dataclass(frozen=True)
class Task(cp.Config):
    """Creates a master spectrum from stacked spectra"""

    #
    # Common information
    #

    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], cp.Param.config(env_var_name="RASSINE_CONFIG")]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, cp.Param.store(cp.parsers.path_parser, env_var_name="RASSINE_ROOT")]

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

    prog_ = Path(__file__).stem

    ini_strict_sections_ = [Path(__file__).stem.split("_")[0]]

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
        "RV_shift": 0,
        "SNR_5500": SNR,
        "lamp_offset": 0,
        "acc_sec": first.acc_sec,
        "berv": BERV,
        "berv_min": BERV_MIN,
        "berv_max": BERV_MAX,
        "instrument": first.instrument,
        "mjd": 0,
        "jdb": 0,
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
