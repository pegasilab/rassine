"""
Implementation of the matching_anchors_filter script
"""
from __future__ import annotations

import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, TypedDict, Union, cast

import configpile as cp
import numpy as np
import tybles as tb
from filelock import FileLock
from numpy.typing import NDArray
from typing_extensions import Annotated

from ..lib.data import LoggingLevel, NameRow, PathPattern, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import local_max, make_continuum
from ..lib.util import log_task_name_and_time
from ..rassine.data import RassineBasicOutput, RassineParameters, RassinePickle


@dataclass(frozen=True)
class MatchingAnchorsRow:
    """
    Describes the effect of the matching_anchors step
    """

    name: str
    is_master: bool
    n_anchors_before: int
    n_anchors_ooc: int
    n_anchors_twin: int
    n_anchors_adding: int

    @staticmethod
    def schema() -> tb.Schema[MatchingAnchorsRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            MatchingAnchorsRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


class MasterToolPickle(TypedDict):
    """Format of the master tool created by the matching_anchors_scan step"""

    curve: NDArray[np.bool_]
    border: NDArray[np.float64]
    wave: NDArray[np.float64]
    threshold: float
    tolerance: float
    fraction: float
    nb_copies_master: int

    #: Warning: only the final path component of the master file is included
    master_filename: Optional[str]


class AnchorParameters(TypedDict):
    #: Master tool file used to compute the matching anchors
    master_tool: str
    master_filename: Optional[str]
    #: Parameters used to define the clusters considered has significant
    threshold: float
    #: Parameters used to define the clusters considered has significant
    tolerance: float
    fraction: float
    #: Number of copies of the master spectra. Such parameter is used to fix the position of the anchors points on the master spectra by overweighting it in the timeseries
    nb_copies_master: int
    #: Continuum called. Usually performed on the 'output' continua. Used later by YARARA to construct the tree of the reduction.
    sub_dico_used: str


class AnchorOutput(TypedDict):
    #: Rassine parameters used to match the anchor points locations
    parameters: AnchorParameters
    #: Rassine fitted continuum
    continuum_linear: NDArray[np.float64]
    #: Wavelength values of the anchors points used (continuum_linear being a linear interpolation between them)
    anchor_wave: NDArray[np.float64]
    #: Flux values of the anchors points used (continuum_linear being a linear interpolation between them)
    anchor_flux: NDArray[np.float64]
    #: Index values of the anchors points used (continuum_linear being a linear interpolation between them)
    anchor_index: NDArray[np.int64]


class AnchorPickle(TypedDict):
    #: Wavelength of the spectrum
    wave: NDArray[np.float64]
    #: Raw flux of the spectrum in photon count units
    flux: NDArray[np.float64]
    #: Raw flux uncertainties of the spectrum in photon count units
    flux_err: Optional[NDArray[np.float64]]
    #: Smoothed raw spectrum by RASSINE used to fit the continuum. Avoid to fit the upper envelop of the noise.
    flux_used: NDArray[np.float64]
    #: Rassine output continuum (individual spectra normalisation)
    output: RassineBasicOutput
    #: Rassine derived parameters
    parameters: RassineParameters
    #: Rassine output continuum (spectra time-series cluster identified)
    matching_anchors: AnchorOutput


class MatchingDiffParameters(TypedDict):
    reference_continuum: str
    savgol_window: int
    recenter: bool
    sub_dico_used: str  # always matching_nachors


class MatchingDiffOutput(TypedDict):
    parameters: MatchingDiffParameters
    continuum_linear: NDArray[np.float64]


class MatchingPickle(TypedDict):
    #: Wavelength of the spectrum
    wave: NDArray[np.float64]
    #: Raw flux of the spectrum in photon count units
    flux: NDArray[np.float64]
    #: Raw flux uncertainties of the spectrum in photon count units
    flux_err: Optional[NDArray[np.float64]]
    #: RASSINE smoothed flux of the spectrum used to fit the continuum (in order to avoid fitting the upper envelop of the noise).
    flux_used: NDArray[np.float64]
    #: Rassine output continuum (individual spectra normalisation)
    output: RassineBasicOutput
    #: Rassine derived parameters
    parameters: RassineParameters
    #: Rassine output continuum (spectra time-series cluster identified)
    matching_anchors: AnchorOutput
    #: Rassine output continuum (spectra time-series cluster identified + low-pass filtering)
    matching_diff: MatchingDiffOutput
