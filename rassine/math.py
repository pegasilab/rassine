from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from .types import Float

c_lum = 299.792e6
h_planck = 6.626e-34
k_boltz = 1.3806e-23


def create_grid(wave_min: Float, dwave: Float, nb_bins: int) -> NDArray[np.float64]:
    """
    Create an equidistant wavelength vector

    Args:
        wave_min: Minimum wavelength value
        dwave: Wavelength step
        nb_bins: Length of the wavelength vector

    Returns:
        The vector containing the equidistant values.
    """

    return np.linspace(
        wave_min, wave_min + (nb_bins - 1) * dwave, nb_bins, dtype=np.float64
    )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)


class DopplerResult(NamedTuple):
    plus_v: NDArray[np.float64]
    minus_v: NDArray[np.float64]


def doppler_r(
    lamb: NDArray[np.float64],
    v: Float,
) -> DopplerResult:
    """
    Relativistic doppler shift of a wavelength by a velocity v in kms.

    Args:
        lamb: Wavelength
        v: Velocity in km/s

    Returns:
        A tuple of shifted wavelengths
    """

    factor = np.sqrt((1 + 1000 * v / c_lum) / (1 - 1000 * v / c_lum))
    lambo = lamb * factor
    lambs = lamb * factor ** (-1)
    return DopplerResult(lambo, lambs)


def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-0.5 * (x - cen) ** 2 / wid**2) + offset
