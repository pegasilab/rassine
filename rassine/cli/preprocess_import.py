from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

import configpile as cp
import numpy as np
import numpy.typing as npt
import tybles as tb
from astropy.io import fits
from filelock import FileLock
from numpy.typing import NDArray
from typing_extensions import Annotated

from rassine.io import save_pickle

from ..analysis import grouping
from ..data import absurd_minus_99_9
from .data import LoggingLevel, PickleProtocol
from .preprocess_table import IndividualBasicRow
from .util import log_task_name_and_time


class PickledIndividualSpectrum(TypedDict):
    """
    Data format of the pickle files produced by the preprocessing step
    """

    #: Wavelength in Angstroms
    wave: npt.NDArray[np.float64]

    #: Flux in photon count units
    flux: npt.NDArray[np.float64]

    #: Error on flux
    flux_err: npt.NDArray[np.float64]

    #: Instrument name
    instrument: str

    #: Observation time in mjd
    mjd: np.float64

    #: Observation time in jdb
    jdb: np.float64

    #: Berv
    berv: np.float64

    #: Simultaneous drift in m/s
    lamp_offset: np.float64

    #: Parallax in milliarcseconds
    plx_mas: np.float64

    #: Secular acceleration
    acc_sec: np.float64

    #: np.min(self.wave)
    wave_min: np.float64

    #: np.max(self.wave)
    wave_max: np.float64

    #: Average delta between bins (note that dlambda is now set by a config parameter)
    dwave: np.float64


# TODO: DACE -> metatable
# TODO: remove the mjd patch

# TODO: rename to Imported
@dataclass(frozen=True)
class IndividualImportedRow:
    """Scalar values from individual pickles tabulated for ease of computation"""

    #: Spectrum name without path and extension
    name: str

    #: Instrument name
    instrument: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction in km/s
    model: np.float64

    #: Median value of model (same for all spectra) in km/s
    rv_mean: np.float64

    #: Difference model - rv_mean in km/s
    rv_shift: np.float64

    jdb: np.float64

    berv: np.float64

    #: Document
    #:
    #: Sometimes called lamp
    lamp_offset: np.float64

    plx_mas: np.float64

    acc_sec: np.float64

    wave_min: np.float64

    wave_max: np.float64

    dwave: np.float64

    hole_left: np.float64

    hole_right: np.float64

    @staticmethod
    def schema() -> tb.Schema[IndividualImportedRow]:
        return tb.schema(
            IndividualImportedRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


# Removed stuff: how to recover mjd from fits / filename
# if mt is not None:
#     mjd = mt.table.loc[mt.table["filename"] == str(file.name), "mjd"].values[0]
# else:
#     try:
#         mjd = header["MJD-OBS"]
#     except KeyError:
#         mjd = Time(file.name.split(".")[1]).mjd


@dataclass(frozen=True)
class Task(cp.Config):
    """Import FITS files into pickle files that can be processed by RASSINE"""

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

    #: Input spectrum table
    input_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-I")]

    #: Output spectrum table
    output_table: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-O")]

    #: Relative path to the folder containing the raw spectra
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

    #: Instrument format of the s1d spectra
    instrument: Annotated[
        str, cp.Param.store(cp.parsers.stripped_str_parser, default_value="HARPS")
    ]

    #: Parallax in mas (no more necessary ?)
    plx_mas: Annotated[
        float,
        cp.Param.store(cp.parsers.float_parser, default_value="0.0"),
    ]


def find_hole(wave: NDArray[np.float64], flux: NDArray[np.float64]) -> Tuple[float, float]:
    null_flux = np.where(flux == 0)[0]  # criterion to detect gap between ccd
    left = absurd_minus_99_9
    right = absurd_minus_99_9
    if len(null_flux) > 0:
        mask = grouping(np.diff(null_flux), 0.5, 0)[-1]
        highest = mask[mask[:, 2].argmax()]
        if highest[2] > 1000:
            left = wave[int(null_flux[highest[0]])]  # store left or -99
            right = wave[int(null_flux[highest[1]])]  # store right or -99
    return (left, right)


def preprocess_fits_harps_coraline_harpn(
    t: Task, row: IndividualBasicRow
) -> IndividualImportedRow:
    """Preprocess one spectrum coming from the HARPS/CORALINE/HARPN instrument"""
    logging.info(f"Processing spectrum {row.name}")

    instrument = t.instrument
    plx_mas = t.plx_mas

    input_file = t.root / t.input_folder / row.raw_filename
    output_file = t.root / t.output_folder / (row.name + ".p")

    logging.debug(f"Reading FITS file {input_file}")
    # Load the FITS file
    header = fits.getheader(input_file)  # load the fits hefluxcannot be read it errors
    data = fits.getdata(input_file)
    spectre = data.astype("float64")  # the flux of your spectrum
    spectre_step = np.round(header["CDELT1"], 8)
    wave_min = np.round(header["CRVAL1"], 8)  # to round float32
    wave_max = np.round(
        header["CRVAL1"] + (len(spectre) - 1) * spectre_step, 8
    )  # to round float32
    wave = np.round(np.linspace(wave_min, wave_max, len(spectre)), 8)

    # cut left and right parts with zero flux
    # and reevaluate wave_min and wave_max
    begin = np.min(np.arange(len(spectre))[spectre > 0])
    end = np.max(np.arange(len(spectre))[spectre > 0])
    wave = wave[begin : end + 1]
    spectre = spectre[begin : end + 1]
    wave_min = np.min(wave)
    wave_max = np.max(wave)

    kw = "ESO"
    if instrument == "HARPN":
        kw = "TNG"

    berv = header["HIERARCH " + kw + " DRS BERV"]
    lamp = header["HIERARCH " + kw + " DRS CAL TH LAMP OFFSET"]
    try:
        pma = header["HIERARCH " + kw + " TEL TARG PMA"] * 1000
        pmd = header["HIERARCH " + kw + " TEL TARG PMD"] * 1000
    except:
        pma = 0
        pmd = 0

    if plx_mas:
        distance_m = 1000.0 / plx_mas * 3.08567758e16
        mu_radps = (
            np.sqrt(pma**2 + pmd**2) * 2 * np.pi / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
        )
        acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
    else:
        acc_sec = 0

    if instrument == "CORALIE":
        if np.mean(spectre) < 100000:
            spectre *= 400780143771.18976  # calibrated with HD8651 2016-12-16 AND 2013-10-24

        spectre /= 1.4e10 / 125**2  # calibrated to match with HARPS SNR

    mjd: np.float64 = row.mjd

    jdb = np.float64(mjd + 0.5)

    hole_left, hole_right = find_hole(wave, spectre)
    if hole_left != absurd_minus_99_9 and hole_right != absurd_minus_99_9:
        logging.info(f"Gap detected in s1d between {hole_left:.2f} and {hole_right:.2f}")

    out: PickledIndividualSpectrum = {
        "wave": wave,
        "flux": spectre,
        "flux_err": np.zeros(len(spectre)),
        "instrument": instrument,
        "mjd": mjd,
        "jdb": jdb,
        "berv": berv,
        "lamp_offset": lamp,
        "plx_mas": np.float64(plx_mas),
        "acc_sec": acc_sec,
        "wave_min": wave_min,
        "wave_max": wave_max,
        "dwave": spectre_step,
    }

    logging.debug(f"Writing pickle file {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(out, f, t.pickle_protocol.level)

    return IndividualImportedRow(
        name=row.name,
        instrument=instrument,
        mjd=mjd,
        model=row.model,
        rv_mean=row.rv_mean,
        rv_shift=row.rv_shift,
        jdb=jdb,
        berv=berv,
        lamp_offset=lamp,
        plx_mas=np.float64(plx_mas),
        acc_sec=acc_sec,
        wave_min=wave_min,
        wave_max=wave_max,
        dwave=spectre_step,
        hole_left=np.float64(hole_left),
        hole_right=np.float64(hole_right),
    )


def preprocess_fits_espresso_express(t: Task, row: IndividualBasicRow) -> IndividualImportedRow:
    instrument = t.instrument
    plx_mas = t.plx_mas
    name = row.raw_filename
    input_file = t.root / t.input_folder / row.raw_filename
    output_file = t.root / t.output_folder / (row.name + ".p")

    header = fits.getheader(input_file)  # load the fits header
    data = fits.getdata(input_file)
    spectre = data["flux"].astype("float64")  # the flux of your spectrum
    spectre_error = data["error"].astype("float64")  # the flux of your spectrum
    grid = data["wavelength_air"].astype(
        "float64"
    )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)
    begin = np.min(np.arange(len(spectre))[spectre > 0])  # remove border spectrum with 0 value
    end = np.max(np.arange(len(spectre))[spectre > 0])  # remove border spectrum with 0 value
    grid = grid[begin : end + 1]
    spectre = spectre[begin : end + 1]
    spectre_error = spectre_error[begin : end + 1]
    wave_min = np.min(grid)
    wave_max = np.max(grid)
    spectre_step = np.mean(np.diff(grid))
    mjd = row.mjd

    kw = "ESO"
    if "HIERARCH TNG QC BERV" in header:
        kw = "TNG"

    berv = np.float64(header["HIERARCH " + kw + " QC BERV"])
    lamp = np.float64(0)  # header['HIERARCH ESO DRS CAL TH LAMP OFFSET'] no yet available
    try:
        pma = header["HIERARCH " + kw + " TEL TARG PMA"] * 1000
        pmd = header["HIERARCH " + kw + " TEL TARG PMD"] * 1000
    except:
        pma = 0
        pmd = 0

    if plx_mas:
        distance_m = 1000.0 / plx_mas * 3.08567758e16
        mu_radps = (
            np.sqrt(pma**2 + pmd**2) * 2 * np.pi / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
        )
        acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
    else:
        acc_sec = 0
    jdb = np.float64(mjd) + 0.5

    hole_left, hole_right = find_hole(grid, spectre)
    if hole_left != absurd_minus_99_9 and hole_right != absurd_minus_99_9:
        logging.info(f"Gap detected in s1d between {hole_left:.2f} and {hole_right:.2f}")

    out: PickledIndividualSpectrum = {
        "wave": grid,
        "flux": spectre,
        "flux_err": spectre_error,
        "instrument": instrument,
        "mjd": mjd,
        "jdb": np.float64(jdb),
        "berv": np.float64(berv),
        "lamp_offset": np.float64(lamp),
        "plx_mas": np.float64(plx_mas),
        "acc_sec": acc_sec,
        "wave_min": wave_min,
        "wave_max": wave_max,
        "dwave": spectre_step,
    }
    save_pickle(output_file, out)

    return IndividualImportedRow(
        name=row.name,
        instrument=instrument,
        mjd=mjd,
        model=row.model,
        rv_mean=row.rv_mean,
        rv_shift=row.rv_shift,
        jdb=jdb,
        berv=berv,
        lamp_offset=lamp,
        plx_mas=np.float64(plx_mas),
        acc_sec=acc_sec,
        wave_min=wave_min,
        wave_max=wave_max,
        dwave=spectre_step,
        hole_left=np.float64(hole_left),
        hole_right=np.float64(hole_right),
    )


@log_task_name_and_time(name="preprocess_import")
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()
    # create output folder if not existing
    (t.root / t.output_folder).mkdir(parents=True, exist_ok=True)
    (t.root / t.output_table).parent.mkdir(parents=True, exist_ok=True)
    tyble = IndividualBasicRow.schema().read_csv(t.root / t.input_table, return_type="Tyble")
    inputs: Sequence[int] = t.inputs

    if not inputs:
        inputs = list(range(len(tyble)))
    rows1: List[IndividualImportedRow] = []
    for i in inputs:
        r = tyble[i]
        if t.instrument in ["ESPRESSO", "EXPRESS"]:
            raise NotImplementedError
            # preprocess_fits_espresso_express(t, r)
        elif t.instrument in ["HARPS", "CORALIE", "HARPN"]:
            rows1.append(preprocess_fits_harps_coraline_harpn(t, r))
        else:
            raise ValueError(f"Instrument {t.instrument} not implemented")

    output_table = t.root / t.output_table
    output_table_lockfile = output_table.with_suffix(output_table.suffix + ".lock")

    logging.debug(f"Appending to output table {output_table}")
    with FileLock(output_table_lockfile):
        df = IndividualImportedRow.schema().from_rows(rows1, return_type="DataFrame")
        df.to_csv(output_table, header=not output_table.exists(), mode="a", index=False)


def cli() -> None:
    run(Task.from_command_line_())
