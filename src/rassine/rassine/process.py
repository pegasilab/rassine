"""Processing code for RASSINE, takes a long time to analyze in Pylance"""
from __future__ import annotations

import logging
import time
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erf
from typing_extensions import assert_never

from ..lib.analysis import find_nearest1, grouping
from ..lib.math import c_lum, create_grid, doppler_r, gaussian, local_max, make_continuum, smooth
from ..stacking.data import MasterPickle, StackedBasicRow, StackedPickle
from .data import ExtraPlotData, RassineBasicOutput, RassineParameters, RassinePickle
from .functions import empty_ccd_gap
from .parsing import Auto, Reg, RegPoly, RegSigmoid, Stretching


def clustering(
    array: np.ndarray, threshold: float, num: int
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Detect and form 1D-cluster on an array. A new cluster is formed once the next vector value is farther than a threshold value.

    Args:
        array: The vector used to create the clustering (1D)
        threshold: Threshold value distance used to define a new cluster.
        num: The minimum number of elements to consider a cluster

    Returns:
        The matrix containing the left index, right index and the length of the 1D cluster
    """
    difference = np.diff(array)
    cluster = difference < threshold
    indice = np.arange(len(cluster))[cluster]
    if sum(cluster):
        j = 0
        border_left = [indice[0]]
        border_right = []
        while j < len(indice) - 1:
            if indice[j] == indice[j + 1] - 1:
                j += 1
            else:
                border_right.append(indice[j])
                border_left.append(indice[j + 1])
                j += 1
        border_right.append(indice[-1])
        border = np.array([border_left, border_right]).T
        border = np.hstack([border, (1 + border[:, 1] - border[:, 0])[:, np.newaxis]])

        kept: List[Any] = []
        for j in range(len(border)):
            if border[j, -1] >= num:
                kept.append(array[border[j, 0] : border[j, 1] + 2])
        return np.array(kept, dtype="object")
    else:
        return [np.array([j]) for j in array]


def ccf_fun(
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


def rassine_process(
    output_filename: str,
    row: Optional[StackedBasicRow],
    data: Union[MasterPickle, StackedPickle],
    synthetic_spectrum: bool,
    random_seed: int,
    par_stretching: Stretching,
    par_vicinity: int,
    par_smoothing_box: Union[Literal["auto"], int],
    par_smoothing_kernel: Literal["rectangular", "gaussian", "savgol", "erf", "hat_exp"],
    par_fwhm_: Union[Literal["auto"], float],
    CCF_mask: str,
    mask_telluric: Sequence[Tuple[float, float]],
    par_R_: Union[Literal["auto"], float],
    par_Rmax_: Union[Literal["auto"], float],
    par_reg_nu: Reg,
    denoising_dist: int,
    count_cut_lim: int,
    count_out_lim: int,
    interpol: Literal["cubic", "linear"],
) -> Tuple[RassinePickle, ExtraPlotData]:
    """Runs the RASSINE processing

    Args:
        output_filename: Output filename, used only to populate the returned dict
        data: Input spectrum
        synthetic_spectrum: See config
        random_seed: Random seed used for synthetic spectrum
        par_stretching: See config
        par_vicinity: See config
        par_smoothing_box: See config
        par_smoothing_kernel: See config
        par_fwhm_: See config
        CCF_mask: See config
        mask_telluric: See config
        par_R_: See config
        par_Rmax_: See config
        par_reg_nu: See config
        denoising_dist: See config
        count_cut_lim: v
        count_out_lim: See config
        interpol: See config

    Returns:
        The processed result
    """

    # =============================================================================
    # TAKE THE DATA
    # =============================================================================

    spectrei = data["flux"]  # the flux of your spectrum
    # the error flux of your spectrum
    spectrei_err: Optional[NDArray[np.float64]] = data.get("flux_err", None)

    def get_grid_and_dgrid_from_pickle(
        d: Union[MasterPickle, StackedPickle]
    ) -> Tuple[NDArray[np.float64], np.float64]:
        return create_grid(d["wave_min"], d["dwave"], len(d["flux"])), d["dwave"]

    grid, dgrid = get_grid_and_dgrid_from_pickle(data)
    RV_sys: np.float64 = data["RV_sys"]

    RV_shift: np.float64 = data["RV_shift"]
    mjd: np.float64 = data["mjd"]
    jdb: np.float64 = data["jdb"]
    hole_left: np.float64 = data["hole_left"]
    hole_right: np.float64 = data["hole_right"]
    berv: np.float64 = data["berv"]
    lamp_offset: np.float64 = data["lamp_offset"]
    acc_sec: np.float64 = data["acc_sec"]
    nb_spectra_stacked: int = data["nb_spectra_stacked"]
    arcfiles: Sequence[str] = data["arcfiles"]

    # Preprocess spectrum
    assert np.all(np.isfinite(grid)), "Grid points must be finite"
    assert np.all(np.isfinite(spectrei)), "Flux values must be finite"

    # clamp value to non-negative
    spectrei[spectrei < 0] = 0
    spectrei = empty_ccd_gap(grid, spectrei, left=hole_left, right=hole_right)

    # =============================================================================
    # LOCAL MAXIMA
    # =============================================================================

    logging.info("RASSINE is beginning the reduction")

    begin = time.time()

    logging.info("Computation of the local maxima : LOADING")

    wave_min = grid[0]
    wave_max = grid[-1]

    def compute_SNR() -> float:
        wave_5500 = int(find_nearest1(grid, 5500)[0])
        continuum_5500 = np.nanpercentile(spectrei[wave_5500 - 50 : wave_5500 + 50], 95)
        SNR_0 = np.sqrt(continuum_5500)
        if np.isnan(SNR_0):
            return -99.0
        else:
            return float(SNR_0)

    SNR_0 = compute_SNR()

    logging.info(f"Spectrum SNR at 5500 : {SNR_0:.0f}")

    def compute_normalisation() -> float:
        len_x = wave_max - wave_min
        len_y = np.max(spectrei) - np.min(spectrei)
        return float(len_y) / float(len_x)  # stretch the y axis to scale the x and y axis

    normalisation = compute_normalisation()
    spectre: NDArray[np.float64] = spectrei / normalisation

    if synthetic_spectrum:
        # to avoid to same value of flux in synthetic spectra
        np.random.seed(random_seed)
        spectre += (
            np.random.randn(len(spectre)) * 1e-5 * np.min(np.diff(spectre)[np.diff(spectre) != 0])
        )

    def cosmic_sigma_clipping() -> None:
        for iteration in range(5):  # k-sigma clipping 5 times
            maxi_roll_fast = np.ravel(
                pd.DataFrame(spectre)
                .rolling(int(100 / dgrid), min_periods=1, center=True)
                .quantile(0.99)
            )
            Q3_fast = np.ravel(
                pd.DataFrame(spectre)
                .rolling(int(5 / dgrid), min_periods=1, center=True)
                .quantile(0.75)
            )  # sigma clipping on 5 \AA range
            Q2_fast = np.ravel(
                pd.DataFrame(spectre)
                .rolling(int(5 / dgrid), min_periods=1, center=True)
                .quantile(0.50)
            )
            IQ_fast = 2 * (Q3_fast - Q2_fast)  # type: ignore
            sup_fast = Q3_fast + 1.5 * IQ_fast
            logging.info(
                f"Number of cosmic peaks removed : {np.sum((spectre > sup_fast) & (spectre > maxi_roll_fast)):.0f}"
            )
            mask = (spectre > sup_fast) & (spectre > maxi_roll_fast)
            for j in range(int(par_vicinity / 2)):
                mask = (
                    mask | np.roll(mask, -j) | np.roll(mask, j)
                )  # supress the peak + the vicinity range
            if sum(mask) == 0:
                break
            spectre[mask] = Q2_fast[mask]

    cosmic_sigma_clipping()

    conversion_fwhm_sig = 10 * wave_min / (2.35 * 3e5)  # 5sigma width in the blue

    def compute_fwhm() -> float:
        mask = np.zeros(len(spectre))
        # by default rolling maxima in a 30 angstrom window
        continuum_right = np.ravel(pd.DataFrame(spectre).rolling(int(30 / dgrid)).quantile(1))
        continuum_left = np.ravel(
            pd.DataFrame(spectre[::-1]).rolling(int(30 / dgrid)).quantile(1)
        )[::-1]
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
        both = np.array([continuum_right, continuum_left])
        continuum = np.min(both, axis=0)

        # smoothing of the envelop 15 anstrom to provide more accurate weight
        continuum = smooth(continuum, int(15 / dgrid), shape="rectangular")

        log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
        log_spectrum = interp1d(
            np.log10(grid),
            spectre / continuum,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )(log_grid)

        if CCF_mask != "master":
            mask_ccf = np.genfromtxt(CCF_mask + ".txt")
            line_center = doppler_r(0.5 * (mask_ccf[:, 0] + mask_ccf[:, 1]), RV_sys)[0]
            distance = np.abs(grid - line_center[:, np.newaxis])
            index_f = np.argmin(distance, axis=1)
            mask = np.zeros(len(spectre))
            mask[index_f] = mask_ccf[:, 2]
            log_mask = interp1d(
                np.log10(grid), mask, kind="linear", bounds_error=False, fill_value="extrapolate"
            )(log_grid)
        else:
            index_f, wave_f, flux_f = produce_line(grid, spectre / continuum)
            keep = (0.5 * (flux_f[:, 1] + flux_f[:, 2]) - flux_f[:, 0]) > 0.2
            flux_f = flux_f[keep]
            wave_f = wave_f[keep]
            index_f = index_f[keep]
            mask = np.zeros(len(spectre))
            mask[index_f[:, 0]] = 0.5 * (flux_f[:, 1] + flux_f[:, 2]) - flux_f[:, 0]
            log_mask = interp1d(
                np.log10(grid), mask, kind="linear", bounds_error=False, fill_value="extrapolate"
            )(log_grid)
            if len(mask_telluric) > 0:
                for j in range(len(mask_telluric)):
                    tellurics = (log_grid > np.log10(mask_telluric[j][0])) & (
                        log_grid < np.log10(mask_telluric[j][1])
                    )
                    log_mask[tellurics] = 0

        vrad, ccf = ccf_fun(log_grid, log_spectrum, log_mask, extended=500)
        ccf = ccf[vrad.argsort()]
        vrad = vrad[vrad.argsort()]
        popt, pcov = curve_fit(gaussian, vrad / 1000, ccf, p0=[0, -0.5, 0.9, 3])
        errors_fit = np.sqrt(np.diag(pcov))
        logging.info(f"[AUTO] FWHM computed from the CCF is about : {popt[-1] * 2.35:.2f} [km/s]")
        if errors_fit[-1] / popt[-1] > 0.2:
            logging.warning(
                "Error on the FWHM of the CCF > 20%! Check the CCF and/or enter you own mask"
            )
            plt.figure(figsize=(10, 6))
            plt.plot(vrad / 1000, ccf, label="CCF")
            plt.plot(
                vrad / 1000,
                gaussian(vrad / 1000, popt[0], popt[1], popt[2], popt[3]),
                label="gaussian fit",
            )
            plt.legend()
            plt.title(
                "Debug graphic : CCF and fit to determine the FWHM\n Check that the fit has correctly converged"
            )
            plt.xlabel("Vrad [km/s]")
            plt.ylabel("CCF")

        return popt[-1] * 2.35

    if par_fwhm_ == "auto":
        par_fwhm = compute_fwhm()
    else:
        par_fwhm = par_fwhm_

    del par_fwhm_

    if par_smoothing_box == "auto":

        def perform_auto_smoothing() -> NDArray[np.float64]:
            # grille en vitesse radiale (unités km/s)
            grid_vrad = (grid - wave_min) / grid * c_lum / 1000
            # new grid equidistant
            grid_vrad_equi = np.linspace(grid_vrad.min(), grid_vrad.max(), len(grid))
            # delta velocity
            dv = np.diff(grid_vrad_equi)[0]
            spectrum_vrad = interp1d(
                grid_vrad, spectre, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )(grid_vrad_equi)

            sp = np.fft.fft(spectrum_vrad)
            # List of frequencies
            freq: NDArray[np.float64] = np.fft.fftfreq(grid_vrad_equi.shape[-1]) / dv  # type: ignore
            sig1 = par_fwhm / 2.35  # fwhm-sigma conversion

            if par_smoothing_kernel == "erf":
                # using the calibration curve calibration
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.00210819, -0.04581559, 0.49444111, -1.78135102]), np.log(SNR_0)
                    )
                )
                alpha2 = np.polyval(np.array([-0.04532947, -0.42650657, 0.59564026]), SNR_0)
            elif par_smoothing_kernel == "hat_exp":
                # using the calibration curve calibration
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.01155214, -0.20085361, 1.34901688, -3.63863408]), np.log(SNR_0)
                    )
                )
                alpha2 = np.polyval(np.array([-0.06031564, -0.45155956, 0.67704286]), SNR_0)
            else:
                raise NotImplementedError
            fourier_center = alpha1 / sig1
            fourier_delta = alpha2 / sig1
            cond = abs(freq) < fourier_center

            if par_smoothing_kernel == "erf":
                # erf function
                fourier_filter = 0.5 * (erf((fourier_center - abs(freq)) / fourier_delta) + 1)
            elif par_smoothing_kernel == "hat_exp":
                # Top hat with an exp
                fourier_filter = cond + (1 - cond) * np.exp(
                    -(abs(freq) - fourier_center) / fourier_delta
                )
            else:
                raise NotImplementedError

            fourier_filter = fourier_filter / fourier_filter.max()

            spectrei_ifft = np.fft.ifft(fourier_filter * (sp.real + 1j * sp.imag))
            spectrei_ifft = np.abs(spectrei_ifft)
            spectre_back = interp1d(
                grid_vrad_equi,
                spectrei_ifft,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(grid_vrad)
            median = np.median(abs(spectre_back - spectre))
            IQ = np.percentile(abs(spectre_back - spectre), 75) - median
            mask_out_fourier = np.where(abs(spectre_back - spectre) > (median + 20 * IQ))[0]
            length_oversmooth = int(1 / fourier_center / dv)
            mask_fourier = np.unique(
                mask_out_fourier
                + np.arange(-length_oversmooth, length_oversmooth + 1, 1)[:, np.newaxis]
            )
            mask_fourier = mask_fourier[(mask_fourier >= 0) & (mask_fourier < len(grid))]
            # supress the smoothing of peak to sharp which create sinc-like wiggle
            spectre_back[mask_fourier] = spectre[mask_fourier]
            # suppression of the border which are at high frequencies
            spectre_back[0 : length_oversmooth + 1] = spectre[0 : length_oversmooth + 1]
            spectre_back[-length_oversmooth:] = spectre[-length_oversmooth:]
            return spectre_back

        spectre = perform_auto_smoothing()
    else:

        def perform_given_smoothing(
            spectre: NDArray[np.float64], parsmooth: int
        ) -> NDArray[np.float64]:
            spectre_back = spectre.copy()
            assert par_smoothing_kernel in ["rectangular", "gaussian", "savgol"]

            spectre = smooth(
                spectre,
                parsmooth,
                shape=cast(Literal["rectangular", "gaussian", "savgol"], par_smoothing_kernel),
            )
            median = np.median(abs(spectre_back - spectre))
            IQ = np.percentile(abs(spectre_back - spectre), 75) - median
            mask_out = np.where(abs(spectre_back - spectre) > (median + 20 * IQ))[0]
            mask_out = np.unique(mask_out + np.arange(-parsmooth, parsmooth + 1, 1)[:, np.newaxis])
            mask_out = mask_out[(mask_out >= 0) & (mask_out < len(grid))]

            # supress the smoothing of peak to sharp which create sinc-like wiggle
            spectre[mask_out.astype("int")] = spectre_back[mask_out.astype("int")]
            return spectre

        spectre = perform_given_smoothing(spectre, par_smoothing_box)

    # conversion of the fwhm to angstrom lengthscale in the bluest part
    par_fwhm = par_fwhm * conversion_fwhm_sig

    spectre = empty_ccd_gap(grid, spectre, left=hole_left, right=hole_right)

    index, flux = local_max(spectre, par_vicinity)
    wave: NDArray[np.float64] = grid[index]
    if flux[0] < spectre[0]:
        wave = np.insert(wave, 0, grid[0])
        flux = np.insert(flux, 0, spectre[0])
        index = np.insert(index, 0, 0)

    if flux[-1] < spectre[-1]:
        wave = np.hstack([wave, grid[-1]])
        flux = np.hstack([flux, spectre[-1]])
        index = np.hstack([index, len(spectre) - 1])

    # supression of cosmic peak
    median: NDArray[np.float64] = np.ravel(
        pd.DataFrame(flux).rolling(10, center=True).quantile(0.50)
    )
    IQ: NDArray[np.float64] = (
        np.ravel(pd.DataFrame(flux).rolling(10, center=True).quantile(0.75)) - median
    )
    IQ[np.isnan(IQ)] = spectre.max()
    median[np.isnan(median)] = spectre.max()
    mask = flux > median + 20 * IQ
    logging.info(f" Number of cosmic peaks removed : {np.sum(mask):.0f}")
    wave = wave[~mask]
    flux = flux[~mask]
    index = index[~mask]

    calib_low = np.polyval([-0.08769286, 5.90699857], par_fwhm / conversion_fwhm_sig)
    calib_high = np.polyval([-0.38532535, 20.17699949], par_fwhm / conversion_fwhm_sig)

    logging.info(
        f"Suggestion of a streching parameter to try : {calib_low + (calib_high - calib_low) * 0.5:.0f} +/- {(calib_high - calib_low) * 0.25:.0f}"
    )

    out_of_calibration = False
    if par_fwhm / conversion_fwhm_sig > 30:
        out_of_calibration = True
        logging.warning("Star out of the FWHM calibration range")

    if isinstance(par_stretching, Auto):
        if not out_of_calibration:
            par_stretching = float(calib_low + (calib_high - calib_low) * par_stretching.ratio)
            logging.info(f"[AUTO] par_stretching fixed : {par_stretching:.2f}")
        else:
            logging.info("[AUTO] par_stretching out of the calibration range, value fixed at 7")
            par_stretching = 7.0

    spectre = spectre / par_stretching
    flux = flux / par_stretching
    normalisation = normalisation * par_stretching

    logging.info(" Computation of the local maxima : DONE")

    loc_max_time = time.time()

    logging.info(f" Time of the step : {loc_max_time - begin:.2f}")

    waves = wave - wave[:, np.newaxis]
    distance = np.sign(waves) * np.sqrt((waves) ** 2 + (flux - flux[:, np.newaxis]) ** 2)
    distance[distance < 0] = 0

    numero = np.arange(len(distance)).astype("int")

    # =============================================================================
    #  PENALITY
    # =============================================================================

    logging.info("Computation of the penality map : LOADING")

    # general parameters for the algorithm
    # (no need to modify the values except if you are visually unsatisfied of the penality plot)
    # iteration increase the upper zone of the penality top

    windows = 10.0  # 10 typical line width scale (small window for the first continuum)
    big_windows = 100.0  # 100 typical line width scale (large window for the second continuum)
    iteration = 5
    reg = par_reg_nu

    if par_R_ == "auto":
        par_R: float = float(np.round(10 * par_fwhm, 1))
        logging.info(f"[AUTO] R fixed : {par_R:.1f}")
        if par_R > 5.0:
            logging.warning("R larger than 5, R fixed at 5")
            par_R = 5.0
    else:
        par_R = par_R_
    del par_R_
    if out_of_calibration:
        windows = 2.0  # 2 typical line width scale (small window for the first continuum)
        big_windows = 20.0  # 20typical line width scale (large window for the second continuum)

    law_chromatic = wave / wave_min

    radius = par_R * np.ones(len(wave)) * law_chromatic
    if (par_Rmax_ != par_R) | (par_Rmax_ == "auto"):
        dx = par_fwhm / np.median(np.diff(grid))
        continuum_small_win = np.ravel(
            pd.DataFrame(spectre).rolling(int(windows * dx), center=True).quantile(1)
        )  # rolling maximum with small windows
        continuum_right = np.ravel(
            pd.DataFrame(spectre).rolling(int(big_windows * dx)).quantile(1)
        )
        continuum_left = np.ravel(
            pd.DataFrame(spectre[::-1]).rolling(int(big_windows * dx)).quantile(1)
        )[::-1]
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
        both = np.array([continuum_right, continuum_left])
        continuum_small_win[
            np.isnan(continuum_small_win) & (2 * grid < (wave_max + wave_min))
        ] = continuum_small_win[~np.isnan(continuum_small_win)][0]
        continuum_small_win[
            np.isnan(continuum_small_win) & (2 * grid > (wave_max + wave_min))
        ] = continuum_small_win[~np.isnan(continuum_small_win)][-1]
        continuum_large_win = np.min(
            both, axis=0
        )  # when taking a large window, the rolling maximum depends on the direction make both direction and take the minimum

        median_large: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.5)
        )
        Q3_large: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        q3_large: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        Q1_large: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        q1_large: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        IQ1_large = Q3_large - Q1_large
        IQ2_large = q3_large - q1_large
        sup_large = np.min([Q3_large + 1.5 * IQ1_large, q3_large + 1.5 * IQ2_large], axis=0)

        mask = continuum_large_win > sup_large
        continuum_large_win[mask] = median_large[mask]

        median_small: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.5)
        )
        Q3_small: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        q3_small: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        Q1_small: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        q1_small: NDArray[np.float64] = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        IQ1_small = Q3_small - Q1_small
        IQ2_small = q3_small - q1_small
        sup_small = np.min([Q3_small + 1.5 * IQ1_small, q3_small + 1.5 * IQ2_small], axis=0)

        mask = continuum_small_win > sup_small
        continuum_small_win[mask] = median_small[mask]

        loc_out = local_max(continuum_large_win, 2)[0]
        for k in loc_out.astype("int"):
            continuum_large_win[k] = np.min(
                [continuum_large_win[k - 1], continuum_large_win[k + 1]]
            )

        loc_out = local_max(continuum_small_win, 2)[0]
        for k in loc_out.astype("int"):
            continuum_small_win[k] = np.min(
                [continuum_small_win[k - 1], continuum_small_win[k + 1]]
            )

        continuum_large_win: NDArray[np.float64] = np.where(
            continuum_large_win == 0, 1.0, continuum_large_win
        )  # replace null values
        penalite0: NDArray[np.float64] = (
            continuum_large_win - continuum_small_win  # type: ignore
        ) / continuum_large_win
        penalite0[penalite0 < 0] = 0
        penalite = penalite0.copy()

        for j in range(
            iteration
        ):  # make the continuum less smooth (step-like function) to improve the speed later
            continuum_right = np.ravel(
                pd.DataFrame(penalite).rolling(int(windows * dx)).quantile(1)
            )
            continuum_left = np.ravel(
                pd.DataFrame(penalite[::-1]).rolling(int(windows * dx)).quantile(1)
            )[::-1]
            continuum_right[np.isnan(continuum_right)] = continuum_right[
                ~np.isnan(continuum_right)
            ][
                0
            ]  # define for the left border all nan value to the first non nan value
            continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][
                -1
            ]  # define for the right border all nan value to the first non nan value
            both = np.array([continuum_right, continuum_left])
            penalite = np.max(both, axis=0)

        penalite_step = penalite.copy()
        mini = penalite_step.min()
        penalite_step = penalite_step - mini
        maxi = penalite_step.max()
        penalite_step = penalite_step / maxi

        penalite_graph = penalite_step[index]
        # take the penalite value at the local maxima position

        threshold = 0.75
        loop = True
        if par_Rmax_ == "auto":
            cluster_length = np.zeros(())
            largest_cluster = -1
            while (loop) & (threshold > 0.2):
                difference = (continuum_large_win < continuum_small_win).astype("int")
                cluster_broad_line = grouping(difference, 0.5, 0)[-1]
                if cluster_broad_line[0][0] == 0:  # rm border left
                    cluster_broad_line = cluster_broad_line[1:]
                if cluster_broad_line[-1][1] == len(grid) - 2:  # rm border right
                    cluster_broad_line = cluster_broad_line[0:-1]

                penality_cluster = np.zeros(len(cluster_broad_line[:, 2]))
                for j in range(len(cluster_broad_line[:, 2])):
                    penality_cluster[j] = np.max(
                        penalite0[cluster_broad_line[j, 0] : cluster_broad_line[j, 1] + 1]
                    )
                cluster_length = np.hstack([cluster_broad_line, penality_cluster[:, np.newaxis]])
                cluster_length = cluster_length[
                    cluster_length[:, 3] > threshold, :
                ]  # only keep cluster with high enough penality
                if len(cluster_length) == 0:
                    threshold -= 0.05
                    continue
                cluster_length = np.hstack(
                    [cluster_length, np.zeros(len(cluster_length))[:, np.newaxis]]
                )
                for j in range(len(cluster_length)):
                    cluster_length[j, 4] = np.nanpercentile(
                        abs(
                            np.diff(spectre[int(cluster_length[j, 0]) : int(cluster_length[j, 1])])
                        ),
                        10,
                    )
                cluster_length = cluster_length[cluster_length[:, 4] != 0, :]
                if len(cluster_length) == 0:
                    threshold -= 0.05
                    continue
                else:
                    loop = False
            if threshold > 0.2:
                band_center = np.mean(grid[cluster_length[:, 0:2].astype("int")], axis=1)
                cluster_length = np.hstack([cluster_length, band_center[:, np.newaxis]])
                largest_cluster = np.argmax(
                    cluster_length[:, 2] / cluster_length[:, 5]
                )  # largest radius in vrad unit
                largest_radius = (
                    cluster_length[largest_cluster, 2] * dgrid
                )  # largest radius in vrad unit

                par_Rmax_ = float(
                    2
                    * np.round(
                        largest_radius
                        * wave_min
                        / cluster_length[largest_cluster, 5]
                        / cluster_length[largest_cluster, 3],
                        0,
                    )
                )
            else:
                if out_of_calibration:
                    par_Rmax_ = 5 * par_R
                else:
                    par_Rmax_ = par_R
            # TOCHECK: removed the if threshold < 0.2 logic
            logging.info(
                f"[AUTO] Rmax found around {cluster_length[largest_cluster, 5]:.0f} AA and fixed : {par_Rmax_:.0f}"
            )
            if par_Rmax_ > 150:  # type: ignore
                logging.warning("Rmax larger than 150, Rmax fixed at 150")
                par_Rmax_ = 150

        par_R = np.round(par_R, 1)
        par_Rmax_ = np.round(par_Rmax_, 1)

        if isinstance(reg, RegPoly):
            radius = law_chromatic * (
                par_R + (float(par_Rmax_) - par_R) * penalite_graph ** float(reg.expo)
            )
        elif isinstance(reg, RegSigmoid):
            radius = law_chromatic * (
                par_R
                + (float(par_Rmax_) - par_R)
                * (1 + np.exp(-reg.steepness * (penalite_graph - reg.center))) ** -1
            )
        else:
            assert_never(reg)

    assert par_Rmax_ != "auto"
    par_Rmax = float(par_Rmax_)
    del par_Rmax_
    logging.info("Computation of the penality map : DONE")

    loc_penality_time = time.time()

    logging.info(f"Time of the step : {loc_penality_time - loc_max_time:.2f}")

    # =============================================================================
    #  ROLLING PIN
    # =============================================================================

    logging.info("Rolling pin is rolling : LOADING")

    mask = (distance > 0) & (distance < 2.0 * par_R)

    count_iter = 0
    # DONE: I removed the loop here
    mask = np.zeros(1)
    radius[0] = radius[0] / 1.5
    keep = [0]
    j = 0
    R_old = par_R

    while len(wave) - j > 3:
        par_R = float(radius[j])  # take the radius from the penality law
        # recompute the points closer than the diameter if Radius changed with the penality
        mask = (distance[j, :] > 0) & (distance[j, :] < 2.0 * par_R)
        while np.sum(mask) == 0:
            par_R *= 1.5
            # recompute the points closer than the diameter if Radius changed with the penality
            mask = (distance[j, :] > 0) & (distance[j, :] < 2.0 * par_R)
        # vector of all the local maxima
        p1 = cast(NDArray[np.float64], np.array([wave[j], flux[j]]).T)
        # vector of all the maxima in the diameter zone
        p2 = cast(NDArray[np.float64], np.array([wave[mask], flux[mask]]).T)
        delta: NDArray[np.float64] = p2 - p1  # delta x delta y
        c = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)  # euclidian distance
        h = np.sqrt(par_R**2 - 0.25 * c**2)
        cx = p1[0] + 0.5 * delta[:, 0] - h / c * delta[:, 1]  # x coordinate of the circles center
        cy = p1[1] + 0.5 * delta[:, 1] + h / c * delta[:, 0]  # y coordinates of the circles center

        cond1 = (cy - p1[1]) >= 0
        thetas = cond1 * (-1 * np.arccos((cx - p1[0]) / par_R) + np.pi) + (1 - 1 * cond1) * (
            -1 * np.arcsin((cy - p1[1]) / par_R) + np.pi
        )
        j2 = thetas.argmin()
        j = numero[mask][j2]  # take the numero of the local maxima falling in the diameter zone
        keep.append(j)
    flux = flux[keep]  # we only keep the local maxima with the rolling pin condition
    wave = wave[keep]
    index = index[keep]

    logging.info(" Rolling pin is rolling : DONE")

    loc_rolling_time = time.time()

    logging.info(f" Time of the step : {loc_rolling_time - loc_penality_time:.2f}")

    # =============================================================================
    # EDGE CUTTING
    # =============================================================================

    logging.info("Edge cutting : LOADING")

    count_cut = 0
    j = 0
    while count_cut < count_cut_lim:
        flux[0 : j + 1] = flux[j + 1]
        flux[-1 - j :] = flux[-2 - j]
        j += 1
        count_cut += 1

    logging.info(" Edge cutting : DONE")

    loc_cutting_time = time.time()

    logging.info(f"Time of the step : {loc_cutting_time - loc_rolling_time:.2f}")

    # =============================================================================
    # CAII MASKING
    # =============================================================================

    mask_caii = ((wave > 3929) & (wave < 3937)) | ((wave > 3964) & (wave < 3972))

    wave = wave[~mask_caii]
    flux = flux[~mask_caii]
    index = index[~mask_caii]

    # =============================================================================
    # OUTLIERS REMOVING
    # =============================================================================

    logging.info("Outliers removing : LOADING")

    count_out = 0
    win_grap = 10  # int(30*typ_line_width/dlambda)
    lines_showed = 5

    while count_out < count_out_lim:
        diff_deri = abs(np.diff(np.diff(flux) / np.diff(wave)))
        mask_out = diff_deri > (np.percentile(diff_deri, 99.5))
        mask_out = np.array([False] + mask_out.tolist() + [False])

        wave = wave[~mask_out]
        flux = flux[~mask_out]
        index = index[~mask_out]
        count_out += 1

    logging.info("Outliers removing : DONE")

    loc_outliers_time = time.time()

    logging.info(f"Time of the step : {loc_outliers_time - loc_cutting_time:.2f}")

    # =============================================================================
    # EQUIDISTANT GRID FORMATION
    # =============================================================================

    len_prev_wave = len(wave)
    criterion = 1
    parameter = np.zeros(())
    for j in range(5):
        diff_x = np.log10(
            np.min(np.vstack([abs(np.diff(wave)[1:]), abs(np.diff(wave)[0:-1])]), axis=0)
        )
        diff_diff_x = np.log10(abs(np.diff(np.diff(wave))) + 1e-5)

        diff_x = np.hstack([0, diff_x, 0])
        diff_diff_x = np.hstack([0, diff_diff_x, 0])
        if criterion == 1:
            parameter = diff_x - diff_diff_x  # type: ignore
        elif criterion == 2:
            parameter = diff_x
        IQ_grid = 2 * (np.nanpercentile(parameter, 50) - np.nanpercentile(parameter, 25))
        mask_out = parameter < (np.nanpercentile(parameter, 50) - 1.5 * IQ_grid)
        if not sum(mask_out):
            criterion += 1
            if criterion == 2:
                continue
            elif criterion == 3:
                break
        mask_out_idx = np.arange(len(parameter))[mask_out]
        if len(mask_out_idx) > 1:
            cluster_idx = clustering(mask_out_idx, 3, 1)
            unique = np.setdiff1d(mask_out_idx, np.hstack(cluster_idx))  # type: ignore
            cluster_idx = list(cluster_idx)
            for j in unique:
                cluster_idx += [np.array([j])]

            mask_out_idx = []
            for j in cluster_idx:
                which = np.argmin(flux[j.astype("int")])
                mask_out_idx.append(j[which])
        mask_out_idx = np.array(mask_out_idx)
        mask_out_idx = list(mask_out_idx[(mask_out_idx > 3) & (mask_out_idx < (len(wave) - 3))])
        mask_out_idx2 = []
        for j in mask_out_idx:
            sub_wave = wave[j - 2 : j + 3]
            sub_diff_diff = []
            for k in [1, 2, 3]:
                sub_diff_diff.append(np.max(abs(np.diff(np.diff(np.delete(sub_wave, k))))))
            mask_out_idx2.append(j - 1 + np.argmin(sub_diff_diff))

        mask_final = np.ones(len(wave)).astype("bool")
        mask_final[mask_out_idx2] = False

        wave = wave[mask_final]
        flux = flux[mask_final]
        index = index[mask_final]

    logging.info(
        f"Number of points removed to build a more equidistant grid : {(len_prev_wave - len(wave))}"
    )

    # =============================================================================
    # PHYSICAL MODEL FITTING (to develop)
    # =============================================================================

    flux_denoised = flux.copy()
    for i, j in enumerate(index):
        if (i < count_cut) | ((len(index) - i) < count_cut):
            pass
        else:
            new = np.mean(spectre[j - denoising_dist : j + denoising_dist + 1])
            if abs(new - flux[i]) / flux[i] < 0.10:
                flux_denoised[i] = new

    flux = flux * normalisation
    flux_denoised = flux_denoised * normalisation

    # =============================================================================
    # FINAL PLOT
    # =============================================================================

    end = time.time()

    logging.info(
        f"[END] RASSINE has finished to compute your continuum in {end - begin:.2f} seconds"
    )

    if interpol == "cubic":
        continuum1, continuum3, continuum1_denoised, continuum3_denoised = make_continuum(
            wave,
            flux,
            flux_denoised,
            grid,
            spectrei,
            continuum_to_produce=(interpol, "undenoised"),
        )
        conti = continuum3
    elif interpol == "linear":
        continuum1, continuum3, continuum1_denoised, continuum3_denoised = make_continuum(
            wave,
            flux,
            flux_denoised,
            grid,
            spectrei,
            continuum_to_produce=(interpol, "undenoised"),
        )
        conti = continuum1

    outputs_interpolation_saved = "linear"
    outputs_denoising_saved = "undenoised"
    continuum1, continuum3, continuum1_denoised, continuum3_denoised = make_continuum(
        wave,
        flux,
        flux_denoised,
        grid,
        spectrei,
        continuum_to_produce=(outputs_interpolation_saved, outputs_denoising_saved),
    )

    # =============================================================================
    # SAVE OF THE PARAMETERS
    # =============================================================================

    if (hole_left is not None) & (hole_left != -99.9):
        hole_left = find_nearest1(grid, doppler_r(hole_left, -30)[0])[1]  # type: ignore
    if (hole_right is not None) & (hole_right != -99.9):
        hole_right = find_nearest1(grid, doppler_r(hole_right, 30)[0])[1]  # type: ignore
    parameters: RassineParameters = {
        "filename": output_filename,
        "number_iteration": count_iter,
        "K_factors": [],
        "axes_stretching": np.round(par_stretching, 1),
        "vicinity_local_max": par_vicinity,
        "smoothing_box": par_smoothing_box,
        "smoothing_kernel": par_smoothing_kernel,
        "fwhm_ccf": float(np.round(par_fwhm / conversion_fwhm_sig, 2)),
        "CCF_mask": CCF_mask,
        "RV_sys": float(RV_sys),
        "min_radius": float(np.round(R_old, 1)),
        "max_radius": float(np.round(par_Rmax, 1)),
        "model_penality_radius": reg.string,
        "denoising_dist": denoising_dist,
        "number_of_cut": count_cut,
        "windows_penality": windows,
        "large_window_penality": big_windows,
        "number_points": len(grid),
        "number_anchors": len(wave),
        "SNR_5500": int(SNR_0),
        "mjd": mjd,
        "jdb": jdb,
        "wave_min": wave_min,
        "wave_max": wave_max,
        "dwave": dgrid,
        "hole_left": hole_left,
        "hole_right": hole_right,
        "RV_shift": RV_shift,
        "berv": berv,
        "lamp_offset": lamp_offset,
        "acc_sec": acc_sec,
        "light_file": True,
        "speedup": 1,
        "continuum_interpolated_saved": outputs_interpolation_saved,
        "continuum_denoised_saved": outputs_denoising_saved,
        "nb_spectra_stacked": nb_spectra_stacked,
        "arcfiles": arcfiles,
        "rv_mean_jdb": None if row is None else row.mean_jdb,
        "rv_dace": None if row is None else row.mean_vrad,
        "rv_dace_std": None if row is None else row.mean_svrad,
    }

    # conversion in fmt format

    flux_used = spectre * normalisation

    # =============================================================================
    # SAVE THE OUTPUT
    # =============================================================================
    basic: RassineBasicOutput = {
        "continuum_linear": continuum1,
        "anchor_wave": wave,
        "anchor_flux": flux,
        "anchor_index": index,
    }

    # we assume light_version=True here
    output: RassinePickle = {
        "wave": grid,
        "flux": spectrei,
        "flux_err": spectrei_err,
        "flux_used": flux_used,
        "output": basic,
        "parameters": parameters,
    }

    return output, ExtraPlotData(normalisation=normalisation, spectre=spectre, conti=conti)
