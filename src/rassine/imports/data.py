from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import tybles as tb
from numpy.typing import NDArray


@dataclass(frozen=True)
class DACE:
    """Columns of the DACE table we import"""

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction km/s
    #:
    #: If not used, set the column values to zero
    model: np.float64

    @staticmethod
    def schema() -> tb.Schema[DACE]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(DACE, order_columns=True, missing_columns="error", extra_columns="drop")

    #: Radial velocity
    vrad: np.float64

    #: Radial velocity uncertainty
    svrad: np.float64

    #: Instrumental drift
    drift_used: np.float64


@dataclass(frozen=True)
class IndividualBasicRow:
    """Columns of the DACE data we extract for the rest of the processing"""

    #: Spectrum name without path and extension
    name: str

    #: Raw filename
    raw_filename: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction in km/s
    model: np.float64

    #: Median value of model (same for all spectra) in km/s
    rv_mean: np.float64

    #: Difference model - rv_mean in km/s
    rv_shift: np.float64

    #: Radial velocity
    vrad: np.float64

    #: Radial velocity uncertainty
    svrad: np.float64

    #: Instrumental drift
    drift: np.float64

    @staticmethod
    def schema() -> tb.Schema[IndividualBasicRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            IndividualBasicRow, order_columns=True, missing_columns="error", extra_columns="drop"
        )


@dataclass(frozen=True)
class IndividualReinterpolatedRow:
    """
    Describes the scalar data associated
    Returns:

    """

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

    SNR_5500: np.float64

    jdb: np.float64

    berv: np.float64

    lamp_offset: np.float64

    plx_mas: np.float64

    acc_sec: np.float64

    # same for all spectra
    hole_left: np.float64

    # same for all spectra
    hole_right: np.float64

    # same for all spectra
    wave_min: np.float64

    # same for all spectra
    wave_max: np.float64

    # same for all spectra
    dwave: np.float64

    # same for all spectra, len(pickled_spectrum.flux)
    nb_bins: np.int64

    #: Radial velocity
    vrad: np.float64

    #: Radial velocity uncertainty
    svrad: np.float64

    #: Instrumental drift
    drift: np.float64

    @staticmethod
    def schema() -> tb.Schema[IndividualReinterpolatedRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            IndividualReinterpolatedRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


class PickledIndividualSpectrum(TypedDict):
    """
    Data format of the pickle files produced by the preprocessing step
    """

    #: Spectrum wavelength in Angstroms
    wave: NDArray[np.float64]

    #: Spectrum flux in photon count units, must not have NaNs
    flux: NDArray[np.float64]

    #: Spectrum flux uncertainties, must not have NaNs
    flux_err: NDArray[np.float64]

    #: Instrument name
    instrument: str

    #: Observation time in mjd
    mjd: np.float64

    #: Observation time in jdb
    jdb: np.float64

    #: Barycentric Earth RV in km/s
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


class ReinterpolatedSpectrumPickle(TypedDict):
    """
    Data format of the pickle files produced by the reinterpolation step
    """

    #: Flux
    flux: NDArray[np.float64]
    #: Flux error
    flux_err: NDArray[np.float64]
    #: Average rv correction (median), same for all spectra
    RV_sys: np.float64
    #: RV correction, shift compared to the median
    RV_shift: np.float64
    #: Corresponds to the square root of the 95th percentile for 100 bins around the wavelength=5500
    SNR_5500: np.float64
    #: what is berv?
    berv: np.float64
    #: what is lamp offset?
    lamp_offset: np.float64
    #: what is plx_mas?
    plx_mas: np.float64
    #: what is acc_sec?
    acc_sec: np.float64
    #: instrument name
    instrument: str
    #: observation time in mjd
    mjd: np.float64
    #: what is jdb?
    jdb: np.float64
    #: Left boundary of hole, or -99.9 if not present
    hole_left: np.float64
    #: Right boundary of hole, or -99.9 if not present
    hole_right: np.float64
    #: Minimum wavelength
    wave_min: np.float64
    # TOCHECK: here
    #: Maximum wavelength, not necessarily equal to np.max(static_grid)
    wave_max: np.float64
    #: delta between two bins, synonym dlambda
    dwave: np.float64


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

    #: Observation date/time in JDB
    jdb: np.float64

    #: Barycentric Earth RV in km/s
    berv: np.float64

    #: Simultaneous drift measurement of the observation in m/s (no more used since 1D spectra are corrected for it)
    lamp_offset: np.float64

    #: Parralax in milliarcseconds (used for secular acceleration correction)
    plx_mas: np.float64

    #: The secular acceleration drift in m/s/year
    acc_sec: np.float64

    #: Wavelength minima wavelength of the spectrum in Angstrom
    wave_min: np.float64

    #: Wavelength maxima value of the spectrum in Angstrom
    wave_max: np.float64

    #: Wavelength step of the spectrum (spectra being evenly sampled in wavelength)
    dwave: np.float64

    #: Wavelength of the left side border for instrument containing hole (HARPS)
    hole_left: np.float64

    #: Wavelength of the right side border for instrument containing hole (HARPS)
    hole_right: np.float64

    #: Radial velocity
    vrad: np.float64

    #: Radial velocity uncertainty
    svrad: np.float64

    #: Instrumental drift
    drift: np.float64

    @staticmethod
    def schema() -> tb.Schema[IndividualImportedRow]:
        """
        Returns the tybles Schema for this dataclass

        :meta private:
        """
        return tb.schema(
            IndividualImportedRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )
