from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

from ...types import Float


class RassineBasicOutput(TypedDict):
    continuum_linear: NDArray[np.float64]
    anchor_wave: NDArray[np.float64]
    anchor_flux: NDArray[np.float64]
    anchor_index: NDArray[np.intp]


class RassineParameters(TypedDict):
    filename: str
    number_iteration: int
    K_factors: Sequence[float]
    axes_stretching: Float
    vicinity_local_max: int
    smoothing_box: Union[Literal["auto"], int]
    smoothing_kernel: Literal["rectangular", "gaussian", "savgol", "erf", "hat_exp"]
    fwhm_ccf: Float
    CCF_mask: str
    RV_sys: Float
    min_radius: Float
    max_radius: Float
    model_penality_radius: str
    denoising_dist: int
    number_of_cut: int
    windows_penality: Float
    large_window_penality: Float
    number_points: int
    number_anchors: int
    SNR_5500: int
    mjd: Float
    jdb: Float
    wave_min: Float
    wave_max: Float
    dwave: Float
    hole_left: Float
    hole_right: Float
    RV_shift: Float
    berv: Float
    lamp_offset: Float
    acc_sec: Float
    light_file: Literal[True]
    speedup: Literal[1]
    continuum_interpolated_saved: Literal["linear"]
    continuum_denoised_saved: Literal["undenoised"]
    nb_spectra_stacked: int
    arcfiles: Sequence[str]


class RassinePickle(TypedDict):
    #: Wavelength
    wave: NDArray[np.float64]
    #: Initial flux
    flux: NDArray[np.float64]
    #: Initial flux error, passed through by RASSINE
    flux_err: Optional[NDArray[np.float64]]
    #: Smoothed flux
    flux_used: NDArray[np.float64]
    #: Rassine output continuum
    output: RassineBasicOutput
    #: Rassine derived parameters
    parameters: RassineParameters


def print_parameters_according_to_paper(parameters: RassineParameters):
    translation: Mapping[str, str] = {
        "number_iteration": "number_iteration",
        "K_factors": "K_factors",
        "axes_stretching": "par_stretching",
        "vicinity_local_max": "par_vicinity",
        "smoothing_box": "par_smoothing_box",
        "smoothing_kernel": "par_smoothing_kernel",
        "fwhm_ccf": "par_fwhm",
        "CCF_mask": "CCF_mask",
        "RV_sys": "RV_sys",
        "min_radius": "par_R",
        "max_radius": "par_Rmax",
        "model_penality_radius": "par_reg_nu",
        "denoising_dist": "denoising_dist",
        "number_of_cut": "count_cut_lim",
        "windows_penality": "windows_penality",
        "large_window_penality": "large_window_penality",
        "number_points": "number of points",
        "number_anchors": "number of anchors",
        "SNR_5500": "SNR_5500",
        "mjd": "mjd",
        "jdb": "jdb",
        "wave_min": "wave_min",
        "wave_max": "wave_max",
        "dwave": "dwave",
        "hole_left": "hole_left",
        "hole_right": "hole_right",
        "RV_shift": "RV_shift",
        "berv": "berv",
        "lamp_offset": "lamp_offset",
        "acc_sec": "acc_sec",
        "light_file": "light_file",
        "speedup": "speedup",
        "continuum_interpolated_saved": "continuum_interpolated_saved",
        "continuum_denoised_saved": "continuum_denoised_saved",
        "nb_spectra_stacked": "nb_spectra_stacked",
        "arcfiles": "arcfiles",
    }
    print("\n------TABLE------- \n")
    for key, display in translation.items():
        print(f"{display} : {parameters[key]}")
    print("\n----------------- \n")
