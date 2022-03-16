import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from configpile import AutoName, Param, Positional, types
from typing_extensions import Annotated

from ..dace import read_dace_csv_and_extract_filename
from ..data import Preprocessed
from ..functions import save_pickle
from .base import RassineConfig, RelPath

# TODO: DACE -> metatable
# TODO: remove the mjd patch
# TODO: check what should be done when FITS are not readable
# TODO: do we need the plx_mas thing?


@dataclass(frozen=True)
class Task(RassineConfig):
    """

    Observation date/time
    """

    ini_strict_sections_ = ["preprocess"]

    #: Relative path to the input data files
    input_folder: Annotated[RelPath, Param.store(RelPath.param_type)]

    #: Input files to process, contains only the file name
    input_files: Annotated[
        Sequence[RelPath],
        Param.append(
            RelPath.param_type.as_sequence_of_one(),
            positional=Positional.ONE_OR_MORE,
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

    #: DACE CSV input file path
    dace_input_file: Annotated[
        Optional[RelPath], Param.store(RelPath.param_type, env_var_name=AutoName.DERIVED)
    ]

    def save(self, pickle_file: Path, output: Preprocessed) -> None:
        with open(pickle_file, "wb") as f:
            pickle.dump(output, f, self.pickle_protocol)

    def preprocess_fits_harps_coraline_harpn(self) -> None:
        table: Optional[pd.DataFrame] = None
        if self.dace_input_file is not None:
            table = read_dace_csv_and_extract_filename(self.root << self.dace_input_file)
        instrument = self.instrument
        plx_mas = self.plx_mas

        for f in self.input_files:
            file = self.root << self.input_folder << f
            name = file.name
            output_file: Path = (self.root << self.output_folder) / (name[:-5] + ".p")

            header = fits.getheader(file)  # load the fits header
            # CHECKME: I have removed the try/catch, now if a file cannot be read it errors
            spectre = fits.getdata(file).astype("float64")  # the flux of your spectrum
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
                    np.sqrt(pma**2 + pmd**2)
                    * 2
                    * np.pi
                    / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
                )
                acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
            else:
                acc_sec = 0

            if instrument == "CORALIE":
                if np.mean(spectre) < 100000:
                    spectre *= (
                        400780143771.18976  # calibrated with HD8651 2016-12-16 AND 2013-10-24
                    )

                spectre /= 1.4e10 / 125**2  # calibrated to match with HARPS SNR

            if table is not None:
                mjd = table.loc[table["filename"] == str(file.name), "mjd"].values[0]
            else:
                try:
                    mjd = header["MJD-OBS"]
                except KeyError:
                    mjd = Time(file.name.split(".")[1]).mjd

            jdb = np.float64(mjd + 0.5)

            out: Preprocessed = {
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

            self.save(output_file, out)

    def preprocess_fits_espresso_express(self) -> None:
        table: Optional[pd.DataFrame] = None
        if self.dace_input_file is not None:
            table = read_dace_csv_and_extract_filename(self.root << self.dace_input_file)
        instrument = self.instrument
        plx_mas = self.plx_mas

        for f in self.input_files:
            file = self.root << self.input_folder << f
            name = file.name
            output_file: Path = (self.root << self.output_folder) / (name[:-5] + ".p")

            header = fits.getheader(file)  # load the fits header
            data = fits.getdata(file)
            # TODO: why not reuse getdata result?
            spectre = data["flux"].astype("float64")  # the flux of your spectrum
            spectre_error = fits.getdata(file)["error"].astype(
                "float64"
            )  # the flux of your spectrum
            grid = data["wavelength_air"].astype(
                "float64"
            )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)
            begin = np.min(
                np.arange(len(spectre))[spectre > 0]
            )  # remove border spectrum with 0 value
            end = np.max(
                np.arange(len(spectre))[spectre > 0]
            )  # remove border spectrum with 0 value
            grid = grid[begin : end + 1]
            spectre = spectre[begin : end + 1]
            spectre_error = spectre_error[begin : end + 1]
            wave_min = np.min(grid)
            wave_max = np.max(grid)
            spectre_step = np.mean(np.diff(grid))
            if table is not None:
                mjd = table.loc[table["filename"] == str(file.name), "mjd"].values[0]
            else:
                try:
                    mjd = float(header["MJD-OBS"])
                except KeyError:
                    mjd = Time(name.split("/")[-1].split(".")[1]).mjd

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
                    np.sqrt(pma**2 + pmd**2)
                    * 2
                    * np.pi
                    / (360.0 * 1000.0 * 3600.0 * 86400.0 * 365.25)
                )
                acc_sec = distance_m * 86400.0 * mu_radps**2  # rv secular drift in m/s per days
            else:
                acc_sec = 0
            jdb = np.array(mjd) + 0.5

            out: Preprocessed = {
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
            self.save(output_file, out)

    # TODO: put back ESPRESSO
    @staticmethod
    def cli() -> None:
        t = Task.from_command_line_()
        assert (t.root << t.output_folder).is_dir(), "The output directory needs to exist"
        # if instrument in ["ESPRESSO", "EXPRESS"]:
        #     preprocess_fits_espresso_express(
        #         inputfiles,
        #         instrument=args.instrument,
        #         plx_mas=args.plx_mas,
        #         output_dir=args.output_dir,
        #     )
        if t.instrument in ["HARPS", "CORALIE", "HARPN"]:
            t.preprocess_fits_harps_coraline_harpn()
        else:
            raise ValueError(f"Instrument {t.instrument} not implemented")
