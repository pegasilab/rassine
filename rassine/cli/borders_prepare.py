import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import deepdish as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from configpile import *
from numpy import float64
from typing_extensions import Annotated

from ..analysis import find_nearest, grouping
from ..io import save_pickle
from ..math import create_grid, doppler_r
from . import preprocess_import
from .data import BasicInfo, LoggingLevel, RassineConfigBeforeStack, RelPath, RootPath

# @dataclass
# class StepDescription:

#     inputs = {"dict1.subdict2": }


class Summary(TypedDict):
    wave_min_k: float
    wave_max_k: float
    dlambda: float
    hole_left_k: float
    hole_right_k: float
    static_grid: Optional[NFArray]  #: size of size of the spectrum?
    wave_min: N  #: vector length is number of spectra
    berv: NFArray  #: vector length is number of spectra
    lamp: NFArray  #: vector length is number of spectra
    plx_mas: NFArray  #: vector length is number of spectra
    acc_sec: NFArray  #: vector length is number of spectra
    rv_shift: NFArray  #: vector length is number of spectra
    rv_mean: Optional[Float]


@dataclass(frozen=True)
class Task(RassineConfigBeforeStack):
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

    #: Relative path to the input data files
    input_folder: Annotated[RelPath, Param.store(RelPath.param_type, short_flag_name="-i")]

    def validate_input_folder_(self) -> Optional[Err]:
        f = self.root.at(self.input_folder)
        return Err.check(f.is_dir(), f"Input folder {f} must exist")

    #: HDF5 output file
    output_file: Annotated[RelPath, Param.store(RelPath.param_type, short_flag_name="-o")]

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


def load(pickle_file: Path) -> preprocess_import.OutputDict:
    with open(pickle_file, "rb") as f:
        return cast(preprocess_import.OutputDict, pickle.load(f))


def run(t: Task) -> Summary:
    """
    Runs the processing
    """

    # from def preprocess_prematch_stellar_frame(files_to_process, rv=0, dlambda=None):
    files: List[Path] = [*(t.root.at(t.input_folder)).glob("*.p")]
    files.sort()
    n = len(files)  # number of files

    assert n > 0, "Must have at least one file"

    mt = Table.read_csv(t.root.at(t.input_master_table), BasicInfo)

    # RV correction
    rv_shift: npt.NDArray[np.float64] = np.zeros((n,))
    rv_mean: Optional[Float] = None
    # CHECKME: what if no rv_shift

    if t.apply_rv_shift:
        assert "model" in mt.table.columns, "A model column must be present in the master table"
        rv = mt.table["model"].to_numpy().astype(np.float64)

        if t.verify_rv_magnitude:
            if np.max(abs(rv)) > 300:
                raise ValueError(
                    "RV value are certainly in m/s instead of km/s!"
                    + "Disable this error using the verify_rv_magnitude parameter"
                )
        rv_mean = np.median(rv)
        rv_shift = rv - rv_mean
        del rv

    wave_min: NFArray = np.zeros((n,))
    wave_max: NFArray = np.zeros((n,))

    hole_left: NFArray = np.empty((0,))
    hole_right: NFArray = np.empty((0,))

    # TODO: diff c'est le step
    diff: NFArray = np.empty((0,))
    all_length: NIArray = np.zeros((n,), dtype=np.int64)
    berv: NFArray = np.zeros((n,))
    lamp: NFArray = np.zeros((n,))
    plx_mas: NFArray = np.zeros((n,))
    acc_sec: NFArray = np.zeros((n,))
    dlambda: Optional[Float] = t.dlambda

    for i, fn in enumerate(files):
        f = load(fn)
        shift = 0.0
        if rv_shift is not None:
            shift = rv_shift[i]

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
                left = wave[int(null_flux[highest[0]])]  # store left or -99
                right = wave[int(null_flux[highest[1]])]  # store right or -99
                hole_left = np.append(hole_left, find_nearest(wave, doppler_r(left, shift)[1])[1])
                hole_right = np.append(
                    hole_right, find_nearest(wave, doppler_r(right, shift)[1])[1]
                )
    hole_left_k: Float = absurd_minus_99_9  # by convention
    hole_right_k: Float = absurd_minus_99_9

    if len(hole_left) != 0:
        hole_left_k = np.min(hole_left) - 0.5  # increase of 0.5 the gap limit by security
        hole_right_k = np.max(hole_right) + 0.5  # increase of 0.5 the gap limit by security
        # CHECKME: f-strings
        logging.warning(f"GAP detected in s1d between {hole_left_k:.2f} and {hole_right_k:.2f}")

    wave_min = np.round(wave_min, 8)  # to take into account float32
    wave_max = np.round(wave_max, 8)  # to take into account float32
    wave_min_k: Float = np.array(wave_min).max()
    wave_max_k: Float = np.array(wave_max).min()
    logging.info(f"Spectra borders are found between : {wave_min_k:.4f} and {wave_max_k:.4f}")

    # TODO: if dlambda is not user given and cannot be determined -> error
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
    assert dlambda is not None  # to make mypy happy
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

    # n-sized: berv, lamp, plx_mas, acc_sec, rv, wave_min (what about wave_max)
    res: Summary = {
        "wave_min_k": wave_min_k,
        "wave_max_k": wave_max_k,
        "dlambda": dlambda,
        "hole_left_k": hole_left_k,
        "hole_right_k": hole_right_k,
        "static_grid": static_grid,
        "wave_min": wave_min,
        "berv": berv,
        "lamp": lamp,
        "plx_mas": plx_mas,
        "acc_sec": acc_sec,
        "rv_shift": rv_shift,
        "rv_mean": rv_mean,
    }
    return res


def cli() -> None:
    t = Task.from_command_line_()
    logging.getLogger().setLevel(t.logging_level)
    data = run(t)
    logging.info(f"Writing borders parameters in {t.root.at(t.output_file)}")
    dd.io.save(t.root.at(t.output_file), data)
