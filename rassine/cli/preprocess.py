import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from configpile import AutoName, Err, Param, Positional, Validator, types
from pydantic.dataclasses import dataclass as pdataclass
from typing_extensions import Annotated

from ..tybles import Row, Table
from .base import BasicInfo, RassineConfigBeforeStack, RelPath

# TODO: DACE -> metatable
# TODO: remove the mjd patch
# TODO: check what should be done when FITS are not readable
# TODO: do we need the plx_mas thing?


class OutputDict(TypedDict):
    """
    Data format of the pickle files produced by the preprocessing step
    """

    #: TODO: doc
    wave: Optional[npt.NDArray[np.float64]]
    #: TODO: doc, size=spectrum length
    flux: npt.NDArray[np.float64]
    #: TODO: doc, size=spectrum length
    flux_err: npt.NDArray[np.float64]
    #: instrument name
    instrument: str
    #: observation time in mjd
    mjd: np.float64
    #: what is jdb?
    jdb: np.float64
    #: what is berv?
    berv: np.float64
    #: what is lamp offset?
    lamp_offset: np.float64
    #: what is plx_mas?
    plx_mas: np.float64
    #: what is acc_sec?
    acc_sec: int
    #: what is wave_min?
    wave_min: np.float64
    #: what is wave_max?
    wave_max: np.float64
    #: what is dwave?
    dwave: np.float64


# Removed stuff: how to recover mjd from fits / filename
# if mt is not None:
#     mjd = mt.table.loc[mt.table["filename"] == str(file.name), "mjd"].values[0]
# else:
#     try:
#         mjd = header["MJD-OBS"]
#     except KeyError:
#         mjd = Time(file.name.split(".")[1]).mjd


@dataclass(frozen=True)
class Task(RassineConfigBeforeStack):
    """

    Observation date/time
    """

    ini_strict_sections_ = ["preprocess"]

    #: Relative path to the input data files
    input_folder: Annotated[RelPath, Param.store(RelPath.param_type)]

    #: Indices of spectrum to process
    inputs: Annotated[
        Sequence[int],
        Param.append(
            types.int_.as_sequence_of_one(),
            positional=Positional.ZERO_OR_MORE,
            long_flag_name=AutoName.FORBIDDEN,
            short_flag_name=None,
        ),
    ]

    #: Instrument format of the s1d spectra
    instrument: Annotated[str, Param.store(types.word, default_value="HARPS")]

    #: Parallax in mas (no more necessary ?)
    plx_mas: Annotated[
        float, Param.store(types.ParamType.from_function_that_raises(float), default_value="0.0")
    ]

    #: Name of the output directory. If None, the output directory is created at the same location than the spectra.
    output_folder: Annotated[
        RelPath,
        Param.store(RelPath.param_type, short_flag_name="-o"),
    ]

    def validate_output_folder_(self) -> Validator:
        return Err.check(
            (self.root << self.output_folder).is_dir(), "The output directory needs to exist"
        )


def save(pickle_file: Path, output: OutputDict, protocol: int) -> None:
    with open(pickle_file, "wb") as f:
        pickle.dump(output, f, protocol)


def preprocess_fits_harps_coraline_harpn(t: Task, row: BasicInfo) -> None:
    instrument = t.instrument
    plx_mas = t.plx_mas
    name = row.filename
    file: Path = (t.root << t.input_folder) / name
    output_file: Path = (t.root << t.output_folder) / (file.stem + ".p")
    header = fits.getheader(file)  # load the fits header
    data = fits.getdata(file)
    # CHECKME: I have removed the try/catch, now if a file cannot be read it errors
    spectre = data.astype("float64")  # the flux of your spectrum
    spectre_step = np.round(header["CDELT1"], 8)
    wave_min = np.round(header["CRVAL1"], 8)  # to round float32
    wave_max = np.round(
        header["CRVAL1"] + (len(spectre) - 1) * spectre_step, 8
    )  # to round float32

    grid = np.round(np.linspace(wave_min, wave_max, len(spectre)), 8)

    begin = np.min(np.arange(len(spectre))[spectre > 0])
    end = np.max(np.arange(len(spectre))[spectre > 0])
    grid = grid[begin : end + 1]
    spectre = spectre[begin : end + 1]
    wave_min = np.min(grid)
    wave_max = np.max(grid)

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

    out: OutputDict = {
        "wave": None,
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

    save(output_file, out, t.pickle_protocol)


def preprocess_fits_espresso_express(t: Task, row: BasicInfo) -> None:
    instrument = t.instrument
    plx_mas = t.plx_mas
    name = row.filename
    file: Path = (t.root << t.input_folder) / name
    output_file: Path = (t.root << t.output_folder) / (file.stem + ".p")

    header = fits.getheader(file)  # load the fits header
    data = fits.getdata(file)
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

    berv = float(header["HIERARCH " + kw + " QC BERV"])
    lamp = 0  # header['HIERARCH ESO DRS CAL TH LAMP OFFSET'] no yet available
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
    jdb = np.array(mjd) + 0.5

    out: OutputDict = {
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
    save(output_file, out, t.pickle_protocol)


def cli() -> None:
    t = Task.from_command_line_()
    mt = Table.read_csv(t.root << t.input_master_table, BasicInfo)
    inputs: Sequence[int] = t.inputs
    if not inputs:
        inputs = list(range(mt.nrows()))
    for i in inputs:
        r = mt[i]
        if t.instrument in ["ESPRESSO", "EXPRESS"]:
            preprocess_fits_espresso_express(t, r)
        elif t.instrument in ["HARPS", "CORALIE", "HARPN"]:
            preprocess_fits_harps_coraline_harpn(t, r)
        else:
            raise ValueError(f"Instrument {t.instrument} not implemented")
