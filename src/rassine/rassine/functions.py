from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..lib.math import Float, doppler_r


def empty_ccd_gap(
    wave: NDArray[np.float64],
    flux: NDArray[np.float64],
    left: Optional[Float] = None,
    right: Optional[Float] = None,
    extended: float = 30.0,
) -> NDArray[np.float64]:
    """
    Ensure a 0 value in the gap between the ccd of HARPS s1d with extended=30 kms extension

    Args:
        wave: Wavelength vector.
        flux: Flux vector.
        left: Wavelength of the left CCF gap.
        right: Wavelength of the right CCF gap.
        extended: Extension of the gap in kms.

    Returns:
        Flux values with null values inside the specified gap
    """

    dgrid = np.diff(wave)[0]

    if left is not None:
        left_ = np.array([left]).astype("float64")
        left = doppler_r(left_, -extended)[0][0]  # 30 km/s supression
    else:
        left = wave.max()

    if right is not None:
        right_ = np.array([right]).astype("float64")
        right = doppler_r(right_, extended)[0][0]  # 30 km/s supression
    else:
        right = wave.min()

    flux[(wave >= left - dgrid / 2) & (wave <= right + dgrid / 2)] = 0
    return flux
