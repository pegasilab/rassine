from typing import Literal, NamedTuple, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import norm
from typing_extensions import TypeAlias, assert_never

absurd_minus_99_9: float = -99.9

Float: TypeAlias = Union[float, np.float64]

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
    """Result of the doppler_r function"""

    #: With positive shift
    plus_v: NDArray[np.float64]

    #: With negative shift
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
    """Gaussian function"""
    return amp * np.exp(-0.5 * (x - cen) ** 2 / wid**2) + offset


def local_max(
    spectre: NDArray[np.float64], vicinity: int
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Perform a local maxima algorithm of a vector.

    Args:
        spectre: The vector to investigate.
        vicinity: The half window in which a local maxima is searched.

    Returns:
        A tuple containing the index and vector values of the detected local maxima.
    """

    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1, vicinity):
        maxima *= (
            0.5
            * (1 + np.sign(vec_base - spectre[vicinity - k : -vicinity - k]))
            * 0.5
            * (1 + np.sign(vec_base - spectre[vicinity + k : -vicinity + k]))
        )

    index = np.where(maxima == 1)[0] + vicinity
    flux = spectre[index]
    return (index, flux)


def smooth(
    y: NDArray[np.float64],
    box_pts: float,
    shape: Literal["rectangular", "gaussian", "savgol"] = "rectangular",
) -> NDArray[np.float64]:  # rectangular kernel for the smoothing

    """
    Smoothing function.

    Args:
        y: The data to be smoothed.
        box_pts: Half-width of the smoothing window.
        shape: The kernel used to smooth the data

    Returns:
        The smoothed data.

    """

    box2_pts = int(2 * box_pts - 1)
    if shape == "savgol":
        if box2_pts >= 5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    else:
        if shape == "rectangular":
            box = np.ones(box2_pts) / box2_pts
        elif shape == "gaussian":
            vec = np.arange(-25, 26)
            box = norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35) / np.sum(
                norm.pdf(vec, scale=(box2_pts - 0.99) / 2.35)
            )
        else:
            assert_never(shape)
        y_smooth = np.convolve(y, box, mode="same")
        y_smooth[0 : int((len(box) - 1) / 2)] = y[0 : int((len(box) - 1) / 2)]
        y_smooth[-int((len(box) - 1) / 2) :] = y[-int((len(box) - 1) / 2) :]
    return y_smooth


def truncated(array, spectre, treshold=5):
    maxi = np.percentile(spectre, 99.9)
    mini = np.percentile(spectre, 0.1)
    tresh = (maxi - mini) / treshold
    array[array < mini - tresh] = mini
    array[array > maxi + tresh] = maxi
    return array


def check_none_negative_values(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Replace negative numbers an a 1-D vector by the local closest non-zero value.

    Args:
        array: The wavelength vector of the spectrum.

    Returns:
        The vector with null values replaced.

    """
    neg = np.where(array <= 0)[0]
    if len(neg) == 0:
        pass
    elif len(neg) == 1:
        array[neg] = 0.5 * (array[neg + 1] + array[neg - 1])
    else:
        where = np.where(np.diff(neg) != 1)[0]
        if (len(where) == 0) & (np.mean(neg) < len(array) / 2):
            array[neg] = array[neg[-1] + 1]
        elif (len(where) == 0) & (np.mean(neg) >= len(array) / 2):
            array[neg] = array[neg[0] - 1]
        else:
            where = np.hstack([where, np.array([0, len(neg) - 1])])
            where = where[where.argsort()]
            for j in range(len(where) - 1):
                if np.mean(array[neg[where[j]] : neg[where[j + 1]] + 1]) < len(array) / 2:
                    array[neg[where[j]] : neg[where[j + 1]] + 1] = array[neg[where[j + 1]] + 1]
                else:
                    array[neg[where[j]] : neg[where[j + 1]] + 1] = array[neg[where[j]] - 1]
    return array


def make_continuum(
    wave,
    flux,
    flux_denoised,
    grid,
    spectrei,
    continuum_to_produce: Tuple[
        Literal["all", "linear", "cubic"], Literal["all", "denoised", "undenoised"]
    ] = ("all", "all"),
):
    """
    Perform the classical sanity check sequence of continuum.

    Parameters
    ----------
    wave : array_like
        Wavelength vector.
    flux : array_like
        Flux vector.
    flux_denoised : array_like
        Flux vector smoothed.
    grid : array_like
        Length of the window for the savgol filtering.
    spectrei : array_like
        No more used ?
    continuum_to_produce : list
        Specifying on which continuum to perform the sanity check.

    Returns
    -------

    continuum_linear : array_like
        The linearly interpolated continuum
    continuum_cubic : array_like
        The cubicly interpolated continuum
    continuum_linear_denoised : array_like
        The linearly interpolated continuum denoised
    continuum_cubic_denoised : array_like
        The cubicly interpolated continuum denoised

    """

    continuum1_denoised = np.zeros(len(grid))
    continuum3_denoised = np.zeros(len(grid))
    continuum1 = np.zeros(len(grid))
    continuum3 = np.zeros(len(grid))

    if continuum_to_produce[1] != "undenoised":
        if continuum_to_produce[0] != "cubic":
            Interpol1 = interp1d(
                wave, flux_denoised, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            continuum1_denoised = Interpol1(grid)
            continuum1_denoised = truncated(continuum1_denoised, spectrei)
            continuum1_denoised = check_none_negative_values(continuum1_denoised)
        if continuum_to_produce[0] != "linear":
            Interpol3 = interp1d(
                wave, flux_denoised, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )
            continuum3_denoised = Interpol3(grid)
            continuum3_denoised = truncated(continuum3_denoised, spectrei)
            continuum3_denoised = check_none_negative_values(continuum3_denoised)

    if continuum_to_produce[1] != "denoised":
        if continuum_to_produce[0] != "cubic":
            Interpol1 = interp1d(
                wave, flux, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            continuum1 = Interpol1(grid)
            continuum1 = truncated(continuum1, spectrei)
            continuum1 = check_none_negative_values(continuum1)
        if continuum_to_produce[0] != "linear":
            Interpol3 = interp1d(
                wave, flux, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )
            continuum3 = Interpol3(grid)
            continuum3 = truncated(continuum3, spectrei)
            continuum3 = check_none_negative_values(continuum3)

    return continuum1, continuum3, continuum1_denoised, continuum3_denoised
