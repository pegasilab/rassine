import glob as glob
from typing import Literal, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import norm

from ..analysis import grouping
from ..io import open_pickle
from ..math import doppler_r
from ..types import Float
from ..util import assert_never
from . import c_lum


def ccf(
    wave: NDArray[np.float64],
    spec1: NDArray[np.float64],
    spec2: NDArray[np.float64],
    extended: int = 1500,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Cross Correlation Function for an equidistant grid in log wavelength.

    Args:
        wave: The wavelength vector of the spectrum.
        spec1: The flux vector of the spectrum.
        spec2: The binary mask used to cross-correlate.
        extended: The length of the extension of the vectors.

    Returns
    -------
    vrad_grid : array_like
        The velocity values of the CCF elements.
    ccf : array_like
        The values of the CCF.

    """

    dwave = wave[1] - wave[0]
    spec1 = np.hstack([np.ones(extended), spec1, np.ones(extended)])
    spec2 = np.hstack([np.zeros(extended), spec2, np.zeros(extended)])
    wave = np.hstack(
        [
            np.arange(-extended * dwave + wave.min(), wave.min(), dwave),
            wave,
            np.arange(wave.max() + dwave, (extended + 1) * dwave + wave.max(), dwave),
        ]
    )  # type: ignore
    shift = np.linspace(0, dwave, 10)[:-1]
    shift_save = []
    sum_spec = np.sum(spec2)
    convolution = []
    for j in shift:
        new_spec = interp1d(
            wave + j, spec2, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )(wave)
        for k in np.arange(-60, 61, 1):
            new_spec2 = np.hstack([new_spec[-k:], new_spec[:-k]])
            convolution.append(np.sum(new_spec2 * spec1) / sum_spec)
            shift_save.append(j + k * dwave)
    return (c_lum * 10 ** np.array(shift_save)) - c_lum, np.array(convolution)


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


def import_files_mcpu_wrapper(args):
    return import_files_mcpu(*args)


def import_files_mcpu(file_list, kind):
    file_list = file_list.tolist()
    sub = []
    snr = []
    for j in file_list:
        file = open_pickle(j)
        snr.append(file["parameters"]["SNR_5500"])  # type: ignore
        sub.append(file["output"][kind])  # type: ignore
    return sub, snr


def postprocess_tofits(path_input, path_rassine, anchor_mode, continuum_mode):

    files_input = glob.glob(path_input)
    files_Rassine = glob.glob(path_rassine)

    files_input.sort()
    files_Rassine.sort()

    for i in range(len(files_input)):

        data_input = fits.open(files_input[i])
        data_Rassine = pd.read_pickle(files_Rassine[i])

        wave_input = data_input[1].data["wavelength"]

        wave_Rassine = data_Rassine["wave"]
        cont_Rassine = data_Rassine[anchor_mode][continuum_mode]

        f = interp1d(
            wave_Rassine, cont_Rassine, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        cont_interp_Rassine = f(wave_input)

        col1 = fits.Column(name="stellar_continuum", format="1D", array=cont_interp_Rassine)
        col2 = fits.Column(
            name="flux_norm", format="1D", array=data_input[1].data["flux"] / cont_interp_Rassine
        )
        cols = fits.ColDefs([col1, col2])

        tbhdu = fits.BinTableHDU.from_columns(data_input[1].data.columns + cols)

        prihdr = fits.Header()
        prihdr = data_input[0].header
        prihdu = fits.PrimaryHDU(header=prihdr)

        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(files_input[i][: files_input[i].index(".fits")] + "_rassine.fits")


def produce_line(
    grid: NDArray[np.float64],
    spectre: NDArray[np.float64],
    box: int = 5,
    shape: Literal["rectangular", "gaussian", "savgol"] = "savgol",
    vic=7,
):
    index, line_flux = local_max(-smooth(spectre, box, shape=shape), vic)
    line_flux = -line_flux
    line_index = index.astype("int")
    line_wave = grid[line_index]

    index2, line_flux2 = local_max(smooth(spectre, box, shape=shape), vic)
    line_index2 = index2.astype("int")
    line_wave2 = grid[line_index2]

    if line_wave[0] < line_wave2[0]:
        line_wave2 = np.insert(line_wave2, 0, grid[0])
        line_flux2 = np.insert(line_flux2, 0, spectre[0])
        line_index2 = np.insert(line_index2, 0, 0)

    if line_wave[-1] > line_wave2[-1]:
        line_wave2 = np.insert(line_wave2, -1, grid[-1])
        line_flux2 = np.insert(line_flux2, -1, spectre[-1])
        line_index2 = np.insert(line_index2, -1, len(grid) - 1)

    memory = np.hstack([-1 * np.ones(len(line_wave)), np.ones(len(line_wave2))])
    stack_wave = np.hstack([line_wave, line_wave2])
    stack_flux = np.hstack([line_flux, line_flux2])
    stack_index = np.hstack([line_index, line_index2])

    memory = memory[stack_wave.argsort()]
    stack_flux = stack_flux[stack_wave.argsort()]
    stack_wave = stack_wave[stack_wave.argsort()]
    stack_index = stack_index[stack_index.argsort()]

    trash, matrix = grouping(memory, 0.01, 0)

    delete_liste = []
    for j in range(len(matrix)):
        number = np.arange(matrix[j, 0], matrix[j, 1] + 2)
        fluxes = stack_flux[number].argsort()
        if trash[j][0] == 1:
            delete_liste.append(number[fluxes[0:-1]])
        else:
            delete_liste.append(number[fluxes[1:]])
    delete_liste = np.hstack(delete_liste)

    memory = np.delete(memory, delete_liste)
    stack_flux = np.delete(stack_flux, delete_liste)
    stack_wave = np.delete(stack_wave, delete_liste)
    stack_index = np.delete(stack_index, delete_liste)

    minima = np.where(memory == -1)[0]
    maxima = np.where(memory == 1)[0]

    index = stack_index[minima]
    index2 = stack_index[maxima]
    flux = stack_flux[minima]
    flux2 = stack_flux[maxima]
    wave = stack_wave[minima]
    wave2 = stack_wave[maxima]

    index = np.hstack([index[:, np.newaxis], index2[0:-1, np.newaxis], index2[1:, np.newaxis]])
    flux = np.hstack([flux[:, np.newaxis], flux2[0:-1, np.newaxis], flux2[1:, np.newaxis]])
    wave = np.hstack([wave[:, np.newaxis], wave2[0:-1, np.newaxis], flux2[1:, np.newaxis]])

    return index, wave, flux


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


def sphinx(sentence: str, rep=None, s2=""):
    answer = "-99.9"
    if rep != None:
        while answer not in rep:
            answer = my_input("Answer : " + s2)
    else:
        answer = my_input("Answer : " + s2)
    return answer


def rm_outliers(
    array: ArrayLike,
    m: float = 1.5,
    kind: Literal["sigma", "inter"] = "sigma",
    direction: Literal["sym", "highest", "lowest"] = "sym",
) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:

    """
    Perform a M-sigma-clipping to remove outliers.

    Args:
        array: The data to filter.
        m: m-sigma value for the clipping.
        kind: The method for the clipping.
              Either 'sigma' (classical std) or 'inter' (IQ sigma clipping)
        direction: In case of interquartile clipping, the direction in which to perform the clipping.
                   Either 'sym','highest' or 'lowest'.

    Returns:
        A tuple (binary_mask, array_filtered) with binary_mask the mask flagging outliers and
        array_filtered the array with the outliers removed
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if kind == "inter":
        median = np.nanpercentile(array, 50)
        Q3 = np.nanpercentile(array, 75)
        Q1 = np.nanpercentile(array, 25)
        IQ = Q3 - Q1
        if direction == "sym":
            mask = (array >= Q1 - m * IQ) & (array <= Q3 + m * IQ)
        elif direction == "highest":
            mask = array <= Q3 + m * 2 * (Q3 - median)
        elif direction == "lowest":
            mask = array >= Q1 - m * 2 * (median - Q1)
        else:
            assert_never(direction)
    elif kind == "sigma":
        mask = abs(array - np.nanmean(array)) <= m * np.nanstd(array)
    else:
        assert_never(kind)
    return mask, array[mask]


def rolling_stat(array, window=1, min_periods=1):
    """
    Perform a rolling statistics.

    Parameters
    ----------
    array : array_like
        The vector to investigate.
    window : int
        The window used for the rolling statistic.
    min_periods: int
        Computation of the statistics up to the min_periods border value

    Returns
    -------
    rolling_median : array_like
        The rolling median.
    rolling_Q1 : array_like
        The rolling 25th percentile.
    rolling_Q3 : array_like
        The rolling 75th percentile.
    rolling_IQ : array_like
        The rolling IQ (Q3-Q1).
    """

    roll_median = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.50)
    )
    roll_Q1 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.25)
    )
    roll_Q3 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.75)
    )
    roll_IQ = roll_Q3 - roll_Q1  # type: ignore
    return roll_median, roll_Q1, roll_Q3, roll_IQ


def truncated(array, spectre, treshold=5):
    maxi = np.percentile(spectre, 99.9)
    mini = np.percentile(spectre, 0.1)
    tresh = (maxi - mini) / treshold
    array[array < mini - tresh] = mini
    array[array > maxi + tresh] = maxi
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
