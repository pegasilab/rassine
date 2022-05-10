from typing import Tuple

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from .types import *

c_lum = 299.792e6


def create_grid(wave_min: float, dwave: float, nb_bins: int) -> NDArray[np.float64]:
    """
    Create an equidistant wavelength vector.

    Args:
        wave_min: Minimum wavelength value.
        dwave: Wavelength step.
        nb_bins: Length of the wavelength vector.

    Returns:
        The vector containing the equidistant values.
    """

    return np.linspace(
        wave_min, wave_min + (nb_bins - 1) * dwave, nb_bins
    )  # the grid of wavelength of your spectrum (assumed equidistant in lambda)


def doppler_r(lamb: float, v: float) -> Tuple[float, float]:
    """
    Relativistic doppler shift of a wavelength by a velocity v in kms.

    Args:
        lamb: Wavelength
        v: Velocity in km/s

    Returns
    -------
    new_wave1:
        Wavelength Doppler shifted with +v
    new_wave2:
        Wavelength Doppler shifted with -v
    """

    factor = np.sqrt((1 + 1000 * v / c_lum) / (1 - 1000 * v / c_lum))
    lambo = lamb * factor
    lambs = lamb * factor ** (-1)
    return lambo, lambs


def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-0.5 * (x - cen) ** 2 / wid**2) + offset
