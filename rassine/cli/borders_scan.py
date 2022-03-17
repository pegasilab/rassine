import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, List, Optional, Sequence, Type, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from attr import s
from configpile import AutoName, Err, Expander, Param, Positional, Result, Validator, types
from configpile.errors import is_err, is_value, wrap_exceptions
from numpy import float64
from typing_extensions import Annotated

from ..analysis import find_nearest, grouping
from ..data import MetaTable, Preprocessed
from ..io import save_pickle
from ..math import create_grid, doppler_r
from ..types import *
from .base import RassineConfig, RelPath


@dataclass(frozen=True)
class Task(RassineConfig):
    """
    Identification tool for wavelength vector parameters

    Define the wavelength vector on which all the spectra will be reinterpolated (common wavelength vector) and detect CDD
    gap.

    The wavelength vector is defined from the maximum of all the spectra minimum wavelength and minimum of all the spectra
    maximum wavelength.

    A RV shift can be applied on all the spectra during the reinterpolation stage.

    It processes all the *.p files present in the input directory.
    """

    ini_strict_sections_ = ["borders-scan"]

    #: Optional DACE CSV input file path
    dace_input_file: Annotated[
        Optional[RelPath],
        Param.store(RelPath.param_type.empty_means_none(), env_var_name=AutoName.DERIVED),
    ]

    def validate_dace_input_file(self) -> Validator:
        if self.dace_input_file is None:
            return None
        fn = self.root << self.dace_input_file
        if not fn.is_file():
            return Err.make(f"Meta table {fn} must exist")
        else:
            return None

    #: Relative path to the input data files
    input_folder: Annotated[RelPath, Param.store(types.path)]

    def validate_input_folder(self) -> Validator:
        folder = self.root << self.input_folder
        if not folder.is_dir():
            return Err.make(f"Input folder {folder} must exist")
        else:
            return None

    #: JSON output file
    output_file: Annotated[RelPath, Param.store(types.path)]

    #: Wavelength sampling step of the spectrum in Angstrom, if not provided, RASSINE attempts guessing
    dlambda: Annotated[
        Optional[float], Param.store(types.float_.empty_means_none(), default_value="")
    ]

    #: Set dlambda to None
    no_dlambda: ClassVar[Expander] = Expander.make("--dlambda", "")

    #: Apply a RV shift in kms to apply on the spectra (e.g binary, large trend), either a float, the path to a pickle file, the path to a CSV file
    apply_rv_shift: Annotated[bool, Param.store(types.bool_, default_value="False")]

    #: Whether to verify that the RV shift is in a reasonable range
    verify_rv_magnitude: Annotated[bool, Param.store(types.bool_, default_value="True")]

    def validate_apply_rv_shift(self) -> Validator:
        if self.apply_rv_shift and self.dace_input_file is None:
            return Err.make(
                "The RV shift is taken from the meta table, so a meta table must be provided"
            )
        else:
            return None

    def load(self, pickle_file: Path) -> Preprocessed:
        with open(pickle_file, "rb") as f:
            return cast(Preprocessed, pickle.load(f))

    def run(self) -> None:

        # from def preprocess_prematch_stellar_frame(files_to_process, rv=0, dlambda=None):
        files_to_process: List[Path] = [*(self.root << self.input_folder).glob("*.p")]
        files_to_process.sort()
        n_files = len(files_to_process)

        # RV correction
        rv_shift: Optional[npt.NDArray[np.float64]] = None

        if self.apply_rv_shift:
            assert self.dace_input_file is not None
            mt = MetaTable.read_csv(self.root << self.dace_input_file)
            assert (
                "model" in mt.table.columns
            ), "A model column must be present in the MetaTable file"
            rv = mt.table["model"].to_numpy().cast(np.float64)
            assert len(rv) == n_files, "Incorrect number of files in meta table"

        if self.verify_rv_magnitude and rv_shift is not None:
            if np.max(abs(rv)) > 300:
                raise ValueError(
                    "RV value are certainly in m/s instead of km/s!"
                    + "Disable this error using the verify_rv_magnitude parameter"
                )

        if rv_shift is not None:
            rv_mean = np.median(rv_shift)
            rv_shift -= rv_mean

        wave_min: NFArray = np.zeros((n_files,))
        wave_max: NFArray = np.zeros((n_files,))

        hole_left: NFArray = np.empty((0,))
        hole_right: NFArray = np.empty((0,))

        # TODO: diff c'est le step
        diff: NFArray = np.empty((0,))
        all_length: NIArray = np.zeros((n_files,), dtype=np.int64)
        berv: NFArray = np.zeros((n_files,))
        lamp: NFArray = np.zeros((n_files,))
        plx_mas: NFArray = np.zeros((n_files,))
        acc_sec: NFArray = np.zeros((n_files,))

        for i, fn in enumerate(files_to_process):
            f = self.load(fn)
            shift = 0.0
            if rv_shift is not None:
                shift = rv[i]

            flux = f["flux"]
            # CHECKME: the logic here
            if f["wave"] is None:
                wave = create_grid(f["wave_min"], f["dwave"], len(flux))
                diff = np.append(diff, f["dwave"])
            else:
                wave = f["wave"]
                if dlambda is None:
                    diff = np.append(diff, np.unique(np.diff(wave)))
            wave_min[i] = f["wave_min"]
            wave_max[i] = f["wave_max"]
            all_length[i] = len(wave)
            berv[i] = f["berv"]
            lamp[i] = f["lamp_offset"]
            plx_mas[i] = f["plx_mas"]
            acc_sec[i] = f["acc_sec"]

            null_flux = np.where(flux == 0)[0]  # criterion to detect gap between ccd
            if len(null_flux):
                mask = grouping(np.diff(null_flux), 0.5, 0)[-1]
                highest = mask[mask[:, 2].argmax()]
                if highest[2] > 1000:
                    left = wave[int(null_flux[highest[0]])]
                    right = wave[int(null_flux[highest[1]])]
                    hole_left = np.append(
                        hole_left, find_nearest(wave, doppler_r(left, shift)[1])[1]
                    )
                    hole_right = np.append(
                        hole_right, find_nearest(wave, doppler_r(right, shift)[1])[1]
                    )
        hole_left_k: Float = absurd_minus_99_9  # by convention
        hole_right_k: Float = absurd_minus_99_9

        if len(hole_left) != 0:
            hole_left_k = np.min(hole_left) - 0.5  # increase of 0.5 the gap limit by security
            hole_right_k = np.max(hole_right) + 0.5  # increase of 0.5 the gap limit by security
            # CHECKME: f-strings
            logging.warning(
                f"GAP detected in s1d between {hole_left_k:.2f} and {hole_right_k:.2f}"
            )

        berv = np.array(berv)
        lamp = np.array(lamp)
        plx_mas = np.array(plx_mas)
        acc_sec = np.array(acc_sec)

        wave_min = np.round(wave_min, 8)  # to take into account float32
        wave_max = np.round(wave_max, 8)  # to take into account float32
        wave_min_k = np.array(wave_min).max()
        wave_max_k = np.array(wave_max).min()
        print(
            "\n [INFO] Spectra borders are found between : %.4f and %.4f"
            % (wave_min_k, wave_max_k)
        )

        # TODO: if dlambda is not user given and cannot be determined -> error
        dlambda: Optional[float] = self.dlambda
        static_grid: Optional[NFArray] = None
        if dlambda is None:
            value: NFArray = np.unique(np.round(diff, 8))
            if len(value) == 1:
                dlambda = value[0]
                logging.info(f"Deduced spectra dwave : {dlambda:.4f}")
            else:
                raise ValueError(
                    "Could not determine the dlambda of your spectral wavelength grid."
                    + " Set the dlambda parameter."
                )
            static_grid = np.arange(wave_min_k, wave_max_k + dlambda, dlambda)

        # CHECKME: verify that the logic is consistent with
        # if dlambda is None:
        #     value = np.unique(np.round(np.hstack(diff),8))

        #     if len(value)==1:
        #         dlambda = value[0]
        #         print('\n [INFO] Spectra dwave is : %.4f \n'%(dlambda))
        #     else:
        #         make_sound('Warning')
        #         print('\n [WARNING] The algorithm has not managed to determine the dlambda value of your spectral wavelength grid')
        #         dlambda = sphinx('Which dlambda value are you selecting for the wavelength grid ?')
        #         dlambda = np.round(np.float(dlambda),8)
        # else:
        #     value = np.array([69,69])

        # if len(value)==1: #case with static wavelength grid
        #     static_grid = None
        # else:
        #     static_grid = np.arange(wave_min_k, wave_max_k + dlambda, dlambda)

        # floats: dlambda, hole_left_k, hole_right_k, rv_mean,wave_min_k, wave_max_k

        # static_grid is separate

        # n-sized: berv, lamp, plx_mas, acc_sec, rv, dlambda, wave_min (what about wave_max)

        return (
            wave_min_k,
            wave_max_k,
            dlambda,
            hole_left_k,
            hole_right_k,
            static_grid,
            wave_min,
            berv,
            lamp,
            plx_mas,
            acc_sec,
            rv,
            rv_mean,
        )


# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     input_dir: Path = args.input_dir
#     dlambda: Optional[float] = args.dlambda
#     rv: Union[npt.ArrayLike, float] = args.rv
#     output_file: Path = args.output_file
#     inputfiles = np.sort(list(input_dir.glob("*.p")))  # type: ignore

#     assert len(inputfiles) > 0, "At least one input file must be available"

#     (
#         wave_min_k,
#         wave_max_k,
#         dlambda,
#         hole_left_k,
#         hole_right_k,
#         static_grid,
#         wave_min,
#         berv,
#         lamp,
#         plx_mas,
#         acc_sec,
#         rv,
#         rv_mean,
#     ) = preprocess_prematch_stellar_frame(list(map(str, inputfiles)), dlambda=dlambda, rv=rv)
#     output = {
#         "wave_min_k": wave_min_k,
#         "wave_max_k": wave_max_k,
#         "dlambda": dlambda,
#         "hole_left_k": hole_left_k,
#         "hole_right_k": hole_right_k,
#         "static_grid": static_grid,
#         "wave_min": wave_min,
#         "berv": berv,
#         "lamp": lamp,
#         "plx_mas": plx_mas,
#         "acc_sec": acc_sec,
#         "rv": rv,
#         "rv_mean": rv_mean,
#     }
#     dd.io.save(output_file, output)
