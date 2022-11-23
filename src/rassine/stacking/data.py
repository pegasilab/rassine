from __future__ import annotations

import typing  # pylint: disable=W0611
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict

import numpy as np
import tybles as tb
from numpy.typing import NDArray
from typing_extensions import Annotated


@dataclass(frozen=True)
class StackedBasicRow:
    """
    Describes the scalar data associated with a stacked spectrum
    """

    #: Stacked spectrum name without path and extension
    name: str
    #: Group index
    group: int
    #: Instrument name
    instrument: str
    #: mjd weighted average
    mjd: np.float64
    #: Average rv correction (median), same for all spectra
    rv_mean: np.float64
    #: RV correction, shift compared to the median, weighted average
    rv_shift: np.float64
    #: Corresponds to the square root of the 95th percentile for 100 bins around the wavelength=5500
    SNR_5500: np.float64
    #: jdb weighted average
    jdb: np.float64
    #: berv, weighted average
    berv: np.float64
    #: min(berv) for the individual spectra in the group
    berv_min: np.float64
    #: max(berv) for the individual spectra in the group
    berv_max: np.float64
    #: lamp_offset, weighted average
    lamp_offset: np.float64
    #: acc_sec, taken from first spectrum
    acc_sec: np.float64
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
    #: Number of days using for the stacking
    stacking_length: np.float64
    #: Number of individual spectra using for this individual spectrum
    nb_spectra_stacked: np.int64
    # same for all spectra, len(flux)
    nb_bins: np.int64

    #: Time weighted by the radial velocity uncertainty (info for YARARA)
    mean_jdb: np.float64

    #: Radial velocity weighted by the radial velocity uncertainty (info for YARARA)
    #:
    #: Median *not* recentered over all averages, YARAR takes care of that
    mean_vrad: np.float64

    #: Propagated harmonic mean of the radial velocity uncertainties (info for YARARA)
    mean_svrad: np.float64

    @staticmethod
    def schema() -> tb.Schema[StackedBasicRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            StackedBasicRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


class StackedPickle(TypedDict):
    """
    Data format of the pickle files produced by the stacking step

    All the weighted averages are made using the bolometric flux
    """

    #: Flux, stacked
    flux: NDArray[np.float64]
    #: Flux error, stacked (square of this is the sum of squares of individual spectra)
    flux_err: NDArray[np.float64]
    #: Average rv correction (median), same for all spectra
    RV_sys: np.float64
    #: RV correction, shift compared to the median, weighted average
    RV_shift: np.float64
    #: Corresponds to the square root of the 95th percentile for 100 bins around the wavelength=5500
    SNR_5500: np.float64
    #: berv, weighted average
    berv: np.float64
    #: lamp_offset, weighted average
    lamp_offset: np.float64
    #: acc_sec, taken from first spectrum
    acc_sec: np.float64
    #: mjd weighted average
    mjd: np.float64
    #: jdb weighted average
    jdb: np.float64
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
    #: Number of days using for the stacking
    stacking_length: np.float64
    #: Number of individual spectra using for this individual spectrum
    nb_spectra_stacked: int
    #: Paths of files used in this stacked spectrum
    arcfiles: Sequence[str]


@dataclass(frozen=True)
class MasterRow:
    """Describes the scalar data associated with the master spectrum"""
    name: str
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
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            MasterRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


@dataclass(frozen=True)
class IndividualGroupRow:
    """Columns of the DACE data we extract for the rest of the processing"""

    #: Spectrum name without path and extension
    name: str

    #: Group index
    group: int

    #: Number of days used for stacking
    stacking_length: np.float64

    #: dbin to shift the binning (0.5 for solar data)
    dbin: np.float64

    @staticmethod
    def schema() -> tb.Schema[IndividualGroupRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            IndividualGroupRow,
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
    flux: NDArray[np.float64]
    #: Tells it is the master spectrum
    master_spectrum: Literal[True]
    #: Average rv correction (median), same for all spectra
    RV_sys: np.float64
    #: RV correction, shift compared to the median, weighted average
    RV_shift: Annotated[np.float64, np.float64(0.0)]
    #: Corresponds to the square root of the 95th percentile for 100 bins around the wavelength=5500
    SNR_5500: np.float64
    #: lamp_offset, weighted average
    lamp_offset: Annotated[np.float64, np.float64(0.0)]
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
    mjd: Annotated[np.float64, np.float64(0.0)]
    #: jdb weighted average
    jdb: Annotated[np.float64, np.float64(0.0)]
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
