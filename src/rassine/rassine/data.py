from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

from ..lib.math import Float


class RassineBasicOutput(TypedDict):
    continuum_linear: NDArray[np.float64]
    anchor_wave: NDArray[np.float64]
    anchor_flux: NDArray[np.float64]
    anchor_index: NDArray[np.int64]


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

    #: Time weighted by the radial velocity uncertainty (info for YARARA)
    #:
    #: Not present in the master spectrum
    rv_mean_jdb: Optional[np.float64]

    #: Std deviation weighted radial velocity average (info for YARARA)
    #:
    #: Comes from the stacking_basic table mean_vrad column (info for YARARA)
    #:
    #: Not present in the master spectrum
    rv_dace: Optional[np.float64]

    #: Radial velocity standard deviation
    #:
    #: Comes from the stacking_basic table mean_svrad column
    #:
    #: Not present in the master spectrum
    rv_dace_std: Optional[np.float64]


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


@dataclass(frozen=True)
class ExtraPlotData:
    normalisation: float
    spectre: NDArray[np.float64]
    conti: NDArray[np.float64]
