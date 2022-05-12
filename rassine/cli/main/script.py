from __future__ import annotations

import dataclasses
import getopt
import logging
import os
import sys
import time
from typing import Literal, Optional, Sequence, Tuple, Union, cast

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import erf
from typing_extensions import Annotated, TypeAlias

from ... import ras
from ...util import assert_never
from ..reinterpolate import PickledReinterpolatedSpectrum
from ..stacking_master_spectrum import MasterPickle
from .config import Auto, Config, RegPoly, RegSigmoid, update_using_anchor_file

# TODO: "auto" -> "erf"
logging.getLogger().setLevel("INFO")


def cli():
    matplotlib.use("Qt5Agg", force=True)

    # get_ipython().run_line_magic('matplotlib','qt5')

    python_version = sys.version[0]

    cfg: Config = Config.from_command_line_()

    spectrum_name = cfg.spectrum_name
    output_dir = cfg.output_dir
    synthetic_spectrum = cfg.synthetic_spectrum

    par_stretching = cfg.par_stretching  # was called axes_stretching
    par_vicinity = cfg.par_vicinity  # was vicinity_local_max
    par_smoothing_box = cfg.par_smoothing_box  # was smoothing_box
    par_smoothing_kernel = cfg.par_smoothing_kernel  # was smoothing_kernel
    par_fwhm = cfg.par_fwhm  # was fwhm_ccf
    CCF_mask = cfg.CCF_mask
    RV_sys = cfg.RV_sys
    mask_telluric = cfg.mask_telluric
    par_R = cfg.par_R  # was min_radius
    par_Rmax = cfg.par_Rmax  # was max_radius
    par_reg_nu = cfg.par_reg_nu  # was model_penality_radius

    denoising_dist = cfg.denoising_dist
    count_cut_lim = cfg.count_cut_lim  # was number_of_cut
    count_out_lim = cfg.count_out_lim  # was number_of_cut_outliers

    interpol = cfg.interpolation  # was interpol
    plot_end = cfg.plot_end
    save_last_plot = cfg.save_last_plot

    outputs_interpolation_saved = cfg.outputs_interpolation_saved  # was outputs_interpolation_save
    outputs_denoising_saved = cfg.outputs_denoising_saved  # was outputs_denoising_saved
    speedup: int = 1

    if cfg.anchor_file is not None:
        assert cfg.anchor_file.exists(), "Anchor file, if provided, must exist"
        update_using_anchor_file(cfg, cfg.anchor_file)

    # DONE: revamped output_dir support, it always mkdirs stuff
    if output_dir is None:
        output_dir = spectrum_name.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # TAKE THE DATA
    # =============================================================================

    plt.close("all")

    # TODO: move to pathlib
    filename = str(spectrum_name).split("/")[-1]
    cut_extension = len(filename.split(".")[-1]) + 1
    new_file = filename[:-cut_extension]

    # TODO: what is that random thing?
    random_number = np.sum([ord(a) for a in filename.split("RASSINE_")[-1]])

    spectrei_err = None
    # CHECKME: we standardize the input format
    data: Union[PickledReinterpolatedSpectrum, MasterPickle] = ras.open_pickle(
        spectrum_name
    )  # load the pickle dictionary

    # TODO: remove the flexibility in dictionary keys
    spectrei = data["flux"]  # the flux of your spectrum
    spectrei_err = data.get("flux_err", None)  # the error flux of your spectrum

    def get_grid_from_pickle(
        d: Union[PickledReinterpolatedSpectrum, MasterPickle]
    ) -> NDArray[np.float64]:
        # TODO: uniform this
        try:
            return d["wave"]  # the grid of wavelength of your spectrum
        except:
            return ras.create_grid(d["wave_min"], d["dwave"], len(d["flux"]))

    grid = get_grid_from_pickle(data)
    # TOCHECK: can this be non?
    # if ras.try_field(data, "RV_sys") is not None:
    RV_sys = data["RV_sys"]

    RV_shift = data["RV_shift"]
    mjd: np.float64 = data["mjd"]
    jdb: np.float64 = data["jdb"]
    hole_left: np.float64 = data["hole_left"]
    hole_right: np.float64 = data["hole_right"]
    berv: np.float64 = data["berv"]
    lamp_offset: np.float64 = data["lamp_offset"]
    acc_sec: np.float64 = data["acc_sec"]
    # TODO: the type of this
    nb_spectra_stacked = ras.try_field(data, "nb_spectra_stacked")
    # TODO: the type of this
    arcfiles = ras.try_field(data, "arcfiles")

    # =============================================================================
    # LOCAL MAXIMA
    # =============================================================================

    logging.info("RASSINE is beginning the reduction")

    begin = time.time()

    logging.info("Computation of the local maxima : LOADING")

    mask_grid = np.arange(len(grid))[(grid - grid) != 0]
    mask_spectre = np.arange(len(grid))[(spectrei - spectrei) != 0]

    if len(mask_grid) > 0:
        print(" Nan values were found, replaced by left and right average...")
        for j in mask_grid:
            grid[j] = (grid[j - 1] + grid[j + 1]) / 2

    if len(mask_spectre) > 0:
        print(" Nan values were found, replaced by left and right average...")
        for j in mask_spectre:
            spectrei[j] = (spectrei[j - 1] + spectrei[j + 1]) / 2

    mask_grid = np.arange(len(grid))[(grid - grid) != 0]
    mask_spectre = np.arange(len(grid))[(spectrei - spectrei) != 0]

    if np.sum(np.isnan(grid)) | np.sum(np.isnan(spectrei)):
        print(" [WARNING] There is too much NaN values, attempting to clean your data")
        spectrei[mask_spectre] = 0

    if len(np.unique(np.diff(grid))) > 1:
        grid_backup_0 = grid.copy()
        new_grid = np.linspace(grid.min(), grid.max(), len(grid))
        spectrei = interp1d(
            grid, spectrei, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )(new_grid)
        grid = new_grid.copy()

    dgrid = grid[1] - grid[0]

    sorting = grid.argsort()  # sort the grid of wavelength
    grid = grid[sorting]
    dlambda = np.mean(np.diff(grid))
    spectrei = spectrei[sorting]
    spectrei[spectrei < 0] = 0
    spectrei = ras.empty_ccd_gap(grid, spectrei, left=hole_left, right=hole_right)

    minx = grid[0]
    maxx = grid[-1]
    miny = np.nanpercentile(spectrei, 0.001)
    maxy = np.nanpercentile(spectrei, 0.999)

    len_x = maxx - minx
    len_y = np.max(spectrei) - np.min(spectrei)

    wave_5500 = int(ras.find_nearest1(grid, 5500)[0])
    continuum_5500 = np.nanpercentile(spectrei[wave_5500 - 50 : wave_5500 + 50], 95)
    SNR_0 = np.sqrt(continuum_5500)
    if np.isnan(SNR_0):
        SNR_0 = -99

    logging.info(f"Spectrum SNR at 5500 : {SNR_0:.0f}")

    normalisation = float(len_y) / float(len_x)  # stretch the y axis to scale the x and y axis
    spectre = spectrei / normalisation

    if synthetic_spectrum:
        spectre += (
            np.random.randn(len(spectre)) * 1e-5 * np.min(np.diff(spectre)[np.diff(spectre) != 0])
        )  # to avoid to same value of flux in synthetic spectra

        # TODO: can be simplified
    # Do the rolling sigma clipping on a grid smaller to increase the speed
    np.random.seed(random_number)
    subset = np.sort(
        np.random.choice(np.arange(len(spectre)), size=int(len(spectre) / 1), replace=False)
    )  # take randomly 1 point over 10 to speed process

    for iteration in range(5):  # k-sigma clipping 5 times
        maxi_roll_fast = np.ravel(
            pd.DataFrame(spectre[subset])
            .rolling(int(100 / dgrid / 1), min_periods=1, center=True)
            .quantile(0.99)
        )
        Q3_fast = np.ravel(
            pd.DataFrame(spectre[subset])
            .rolling(int(5 / dgrid / 1), min_periods=1, center=True)
            .quantile(0.75)
        )  # sigma clipping on 5 \AA range
        Q2_fast = np.ravel(
            pd.DataFrame(spectre[subset])
            .rolling(int(5 / dgrid / 1), min_periods=1, center=True)
            .quantile(0.50)
        )
        Q1_fast = np.ravel(
            pd.DataFrame(spectre[subset])
            .rolling(int(5 / dgrid / 1), min_periods=1, center=True)
            .quantile(0.25)
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

    conversion_fwhm_sig = 10 * minx / (2.35 * 3e5)  # 5sigma width in the blue

    if par_fwhm == "auto":

        mask = np.zeros(len(spectre))
        continuum_right = np.ravel(
            pd.DataFrame(spectre).rolling(int(30 / dgrid)).quantile(1)
        )  # by default rolling maxima in a 30 angstrom window
        continuum_left = np.ravel(
            pd.DataFrame(spectre[::-1]).rolling(int(30 / dgrid)).quantile(1)
        )[::-1]
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
        both = np.array([continuum_right, continuum_left])
        continuum = np.min(both, axis=0)

        continuum = ras.smooth(
            continuum, int(15 / dgrid), shape="rectangular"
        )  # smoothing of the envelop 15 anstrom to provide more accurate weight

        log_grid = np.linspace(np.log10(grid).min(), np.log10(grid).max(), len(grid))
        log_spectrum = interp1d(
            np.log10(grid),
            spectre / continuum,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )(log_grid)

        if CCF_mask != "master":
            # TODO: mask_harps should be mask_ccf
            mask_harps = np.genfromtxt(CCF_mask + ".txt")
            line_center = ras.doppler_r(0.5 * (mask_harps[:, 0] + mask_harps[:, 1]), RV_sys)[0]
            distance = np.abs(grid - line_center[:, np.newaxis])
            index = np.argmin(distance, axis=1)
            mask = np.zeros(len(spectre))
            mask[index] = mask_harps[:, 2]
            log_mask = interp1d(
                np.log10(grid), mask, kind="linear", bounds_error=False, fill_value="extrapolate"
            )(log_grid)
        else:
            index, wave, flux = ras.produce_line(grid, spectre / continuum)
            keep = (0.5 * (flux[:, 1] + flux[:, 2]) - flux[:, 0]) > 0.2
            flux = flux[keep]
            wave = wave[keep]
            index = index[keep]
            mask = np.zeros(len(spectre))
            mask[index[:, 0]] = 0.5 * (flux[:, 1] + flux[:, 2]) - flux[:, 0]
            log_mask = interp1d(
                np.log10(grid), mask, kind="linear", bounds_error=False, fill_value="extrapolate"
            )(log_grid)
            if len(mask_telluric) > 0:
                for j in range(len(mask_telluric)):
                    tellurics = (log_grid > np.log10(mask_telluric[j][0])) & (
                        log_grid < np.log10(mask_telluric[j][1])
                    )
                    log_mask[tellurics] = 0

        vrad, ccf = ras.ccf(log_grid, log_spectrum, log_mask, extended=500)
        ccf = ccf[vrad.argsort()]
        vrad = vrad[vrad.argsort()]
        popt, pcov = curve_fit(ras.gaussian, vrad / 1000, ccf, p0=[0, -0.5, 0.9, 3])
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
                ras.gaussian(vrad / 1000, popt[0], popt[1], popt[2], popt[3]),
                label="gaussian fit",
            )
            plt.legend()
            plt.title(
                "Debug graphic : CCF and fit to determine the FWHM\n Check that the fit has correctly converged"
            )
            plt.xlabel("Vrad [km/s]")
            plt.ylabel("CCF")

        par_fwhm = popt[-1] * 2.35

    if par_smoothing_kernel == "rectangular":
        active_b = 0
    elif par_smoothing_kernel == "gaussian":
        active_b = 1
    elif par_smoothing_kernel == "savgol":
        active_b = 2
    else:
        raise NotImplementedError  # should not happen

    if True:
        # TODO: can remove this, as this is validated during parsing
        if par_smoothing_box == "auto":
            grid_vrad = (
                (grid - minx) / grid * ras.c_lum / 1000
            )  # grille en vitesse radiale (unités km/s)
            grid_vrad_equi = np.linspace(
                grid_vrad.min(), grid_vrad.max(), len(grid)
            )  # new grid equidistant
            dv = np.diff(grid_vrad_equi)[0]  ##delta velocity
            spectrum_vrad = interp1d(
                grid_vrad, spectre, kind="cubic", bounds_error=False, fill_value="extrapolate"
            )(grid_vrad_equi)

            sp = np.fft.fft(spectrum_vrad)
            freq = np.fft.fftfreq(grid_vrad_equi.shape[-1]) / dv  # List of frequencies
            sig1 = par_fwhm / 2.35  # fwhm-sigma conversion

            if par_smoothing_kernel == "erf":
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.00210819, -0.04581559, 0.49444111, -1.78135102]), np.log(SNR_0)
                    )
                )  # using the calibration curve calibration
                alpha2 = np.polyval(np.array([-0.04532947, -0.42650657, 0.59564026]), SNR_0)
            elif par_smoothing_kernel == "hat_exp":
                alpha1 = np.exp(
                    np.polyval(
                        np.array([0.01155214, -0.20085361, 1.34901688, -3.63863408]), np.log(SNR_0)
                    )
                )  # using the calibration curve calibration
                alpha2 = np.polyval(np.array([-0.06031564, -0.45155956, 0.67704286]), SNR_0)
            else:
                raise NotImplementedError
            fourier_center = alpha1 / sig1
            fourier_delta = alpha2 / sig1
            cond = abs(freq) < fourier_center

            if par_smoothing_kernel == "erf":
                fourier_filter = 0.5 * (
                    erf((fourier_center - abs(freq)) / fourier_delta) + 1
                )  # erf function
                smoothing_shape = "erf"
            elif par_smoothing_kernel == "hat_exp":
                fourier_filter = cond + (1 - cond) * np.exp(
                    -(abs(freq) - fourier_center) / fourier_delta
                )  # Top hat with an exp
                smoothing_shape = "hat_exp"
            else:
                raise NotImplementedError

            fourier_filter = fourier_filter / fourier_filter.max()

            spectrei_ifft = np.fft.ifft(fourier_filter * (sp.real + 1j * sp.imag))
            # spectrei_ifft *= spectre.max()/spectrei_ifft.max()
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
            # plt.plot(grid_vrad_equi,abs(spectrei_ifft-spectrum_vrad))
            # plt.axhline(y=median+20*IQ)
            length_oversmooth = int(1 / fourier_center / dv)
            mask_fourier = np.unique(
                mask_out_fourier
                + np.arange(-length_oversmooth, length_oversmooth + 1, 1)[:, np.newaxis]
            )
            mask_fourier = mask_fourier[(mask_fourier >= 0) & (mask_fourier < len(grid))]
            spectre_back[mask_fourier] = spectre[
                mask_fourier
            ]  # supress the smoothing of peak to sharp which create sinc-like wiggle
            spectre_back[0 : length_oversmooth + 1] = spectre[
                0 : length_oversmooth + 1
            ]  # suppression of the border which are at high frequencies
            spectre_back[-length_oversmooth:] = spectre[-length_oversmooth:]
            spectre = spectre_back.copy()
            smoothing_length = par_smoothing_box
        else:
            spectre_backup = spectre.copy()
            assert par_smoothing_kernel in ["rectangular", "gaussian", "savgol"]

            spectre = ras.smooth(
                spectre,
                int(par_smoothing_box),
                shape=cast(Literal["rectangular", "gaussian", "savgol"], par_smoothing_kernel),
            )
            smoothing_shape = par_smoothing_kernel
            smoothing_length = par_smoothing_box
            median = np.median(abs(spectre_backup - spectre))
            IQ = np.percentile(abs(spectre_backup - spectre), 75) - median
            mask_out = np.where(abs(spectre_backup - spectre) > (median + 20 * IQ))[0]
            mask_out = np.unique(
                mask_out + np.arange(-smoothing_length, smoothing_length + 1, 1)[:, np.newaxis]
            )
            mask_out = mask_out[(mask_out >= 0) & (mask_out < len(grid))]
            spectre[mask_out.astype("int")] = spectre_backup[
                mask_out.astype("int")
            ]  # supress the smoothing of peak to sharp which create sinc-like wiggle

    par_fwhm = (
        par_fwhm * conversion_fwhm_sig
    )  # conversion of the fwhm to angstrom lengthscale in the bluest part

    spectre = ras.empty_ccd_gap(grid, spectre, left=hole_left, right=hole_right)

    index, flux = ras.local_max(spectre, par_vicinity)
    index = index.astype("int")
    wave = grid[index]

    if flux[0] < spectre[0]:
        wave = np.insert(wave, 0, grid[0])
        flux = np.insert(flux, 0, spectre[0])
        index = np.insert(index, 0, 0)

    if flux[-1] < spectre[-1]:
        wave = np.hstack([wave, grid[-1]])
        flux = np.hstack([flux, spectre[-1]])
        index = np.hstack([index, len(spectre) - 1])

    # supression of cosmic peak
    median = np.ravel(pd.DataFrame(flux).rolling(10, center=True).quantile(0.50))
    IQ = np.ravel(pd.DataFrame(flux).rolling(10, center=True).quantile(0.75)) - median
    # plt.plot(wave,np.ravel(pd.DataFrame(flux).rolling(10,center=True).quantile(0.50))+10*IQ,color='k')
    # plt.scatter(wave,flux)
    IQ[np.isnan(IQ)] = spectre.max()
    median[np.isnan(median)] = spectre.max()
    mask = flux > median + 20 * IQ
    # plt.show()
    logging.info(f" Number of cosmic peaks removed : {np.sum(mask):.0f}")
    wave = wave[~mask]
    flux = flux[~mask]
    index = index[~mask]

    # print(' Rough estimation of the typical width of the lines : median=%.3f mean=%.3f'%(np.median(np.diff(wave))/conversion_fwhm_sig,np.mean(np.diff(wave))/conversion_fwhm_sig))

    computed_parameters = (
        0.390 / 51.3 * np.median(abs(np.diff(flux))) / np.median(np.diff(wave))
    )  # old calibration

    calib_low = np.polyval([-0.08769286, 5.90699857], par_fwhm / conversion_fwhm_sig)
    calib_high = np.polyval([-0.38532535, 20.17699949], par_fwhm / conversion_fwhm_sig)

    logging.info(
        f" Suggestion of a streching parameter to try : {calib_low + (calib_high - calib_low) * 0.5:.0f} +/- {(calib_high - calib_low) * 0.25:.0f}"
    )

    out_of_calibration = False
    if par_fwhm / conversion_fwhm_sig > 30:
        out_of_calibration = True
        print(" [WARNING] Star out of the FWHM calibration range")

    if isinstance(par_stretching, Auto):
        if not out_of_calibration:
            par_stretching = float(calib_low + (calib_high - calib_low) * par_stretching.ratio)
            # TODO: what about this
            # par_stretching = 20*computed_parameters #old calibration
            logging.info(f" [AUTO] par_stretching fixed : {par_stretching:.2f}")
        else:
            print(" [AUTO] par_stretching out of the calibration range, value fixed at 7")
            par_stretching = 7.0

    spectre = spectre / par_stretching
    flux = flux / par_stretching
    normalisation = normalisation * par_stretching

    locmaxx = wave.copy()
    locmaxy = flux.copy()
    locmaxz = index.copy()

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

    np.random.seed(random_number + 1)
    subset = np.sort(
        np.random.choice(np.arange(len(spectre)), size=int(len(spectre) / speedup), replace=False)
    )  # take randomly 1 point over 10 to speed process

    windows = 10.0  # 10 typical line width scale (small window for the first continuum)
    big_windows = 100.0  # 100 typical line width scale (large window for the second continuum)
    iteration = 5
    reg = par_reg_nu
    par_model = reg.name
    Penalty = False

    if par_R == "auto":
        par_R = np.round(10 * par_fwhm, 1)
        logging.info(f"[AUTO] R fixed : {par_R:.1f}")
        if par_R > 5:
            logging.warning("R larger than 5, R fixed at 5")
            par_R = 5

    if out_of_calibration:
        windows = 2.0  # 2 typical line width scale (small window for the first continuum)
        big_windows = 20.0  # 20typical line width scale (large window for the second continuum)

    law_chromatic = wave / minx

    radius = par_R * np.ones(len(wave)) * law_chromatic
    if (par_Rmax != par_R) | (par_Rmax == "auto"):
        Penalty = True
        dx = par_fwhm / np.median(np.diff(grid))

        continuum_small_win = np.ravel(
            pd.DataFrame(spectre[subset])
            .rolling(int(windows * dx / speedup), center=True)
            .quantile(1)
        )  # rolling maximum with small windows
        continuum_right = np.ravel(
            pd.DataFrame(spectre[subset]).rolling(int(big_windows * dx / speedup)).quantile(1)
        )
        continuum_left = np.ravel(
            pd.DataFrame(spectre[subset][::-1])
            .rolling(int(big_windows * dx / speedup))
            .quantile(1)
        )[::-1]
        continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0]
        continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
        both = np.array([continuum_right, continuum_left])
        continuum_small_win[
            np.isnan(continuum_small_win) & (2 * grid[subset] < (maxx + minx))
        ] = continuum_small_win[~np.isnan(continuum_small_win)][0]
        continuum_small_win[
            np.isnan(continuum_small_win) & (2 * grid[subset] > (maxx + minx))
        ] = continuum_small_win[~np.isnan(continuum_small_win)][-1]
        continuum_large_win = np.min(
            both, axis=0
        )  # when taking a large window, the rolling maximum depends on the direction make both direction and take the minimum

        median_large = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.5)
        )
        Q3_large = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        q3_large = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.75)
        )
        Q1_large = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(10 * big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        q1_large = np.ravel(
            pd.DataFrame(continuum_large_win)
            .rolling(int(big_windows * dx), min_periods=1, center=True)
            .quantile(0.25)
        )
        IQ1_large = Q3_large - Q1_large
        IQ2_large = q3_large - q1_large
        sup_large = np.min([Q3_large + 1.5 * IQ1_large, q3_large + 1.5 * IQ2_large], axis=0)

        if speedup > 1:
            sup_large = interp1d(subset, sup_large, bounds_error=False, fill_value="extrapolate")(
                np.arange(len(spectre))
            )
            continuum_large_win = interp1d(
                subset, continuum_large_win, bounds_error=False, fill_value="extrapolate"
            )(np.arange(len(spectre)))
            median_large = interp1d(
                subset, median_large, bounds_error=False, fill_value="extrapolate"
            )(np.arange(len(spectre)))

        mask = continuum_large_win > sup_large
        continuum_large_win[mask] = median_large[mask]

        median_small = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx / speedup), min_periods=1, center=True)
            .quantile(0.5)
        )
        Q3_small = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx / speedup), min_periods=1, center=True)
            .quantile(0.75)
        )
        q3_small = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(big_windows * dx / speedup), min_periods=1, center=True)
            .quantile(0.75)
        )
        Q1_small = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(10 * big_windows * dx / speedup), min_periods=1, center=True)
            .quantile(0.25)
        )
        q1_small = np.ravel(
            pd.DataFrame(continuum_small_win)
            .rolling(int(big_windows * dx / speedup), min_periods=1, center=True)
            .quantile(0.25)
        )
        IQ1_small = Q3_small - Q1_small
        IQ2_small = q3_small - q1_small
        sup_small = np.min([Q3_small + 1.5 * IQ1_small, q3_small + 1.5 * IQ2_small], axis=0)

        if speedup > 1:
            sup_small = interp1d(subset, sup_small, bounds_error=False, fill_value="extrapolate")(
                np.arange(len(spectre))
            )
            continuum_small_win = interp1d(
                subset, continuum_small_win, bounds_error=False, fill_value="extrapolate"
            )(np.arange(len(spectre)))
            median_small = interp1d(
                subset, median_small, bounds_error=False, fill_value="extrapolate"
            )(np.arange(len(spectre)))

        mask = continuum_small_win > sup_small
        continuum_small_win[mask] = median_small[mask]

        loc_out = ras.local_max(continuum_large_win, 2)[0]
        for k in loc_out.astype("int"):
            continuum_large_win[k] = np.min(
                [continuum_large_win[k - 1], continuum_large_win[k + 1]]
            )

        loc_out = ras.local_max(continuum_small_win, 2)[0]
        for k in loc_out.astype("int"):
            continuum_small_win[k] = np.min(
                [continuum_small_win[k - 1], continuum_small_win[k + 1]]
            )

        continuum_large_win = np.where(
            continuum_large_win == 0, 1.0, continuum_large_win
        )  # replace null values
        penalite0 = (continuum_large_win - continuum_small_win) / continuum_large_win
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
        if par_Rmax == "auto":
            cluster_length = np.zeros(())  # TODO: added
            while (loop) & (threshold > 0.2):
                difference = (continuum_large_win < continuum_small_win).astype("int")
                cluster_broad_line = ras.grouping(difference, 0.5, 0)[-1]
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

                par_Rmax = 2 * np.round(
                    largest_radius
                    * minx
                    / cluster_length[largest_cluster, 5]
                    / cluster_length[largest_cluster, 3],
                    0,
                )
            else:
                if out_of_calibration:
                    par_Rmax = 5 * par_R
                else:
                    par_Rmax = par_R
            # TOCHECK: removed the if threshold < 0.2 logic
            logging.info(
                f" [AUTO] Rmax found around {cluster_length[largest_cluster, 5]:.0f} AA and fixed : {par_Rmax:.0f}"
            )
            if par_Rmax > 150:
                logging.warning(" [WARNING] Rmax larger than 150, Rmax fixed at 150")
                par_Rmax = 150

        par_R = np.round(par_R, 1)
        par_Rmax = np.round(par_Rmax, 1)
        # TODO: par_model role
        par_model = reg
        if isinstance(reg, RegPoly):
            radius = law_chromatic * (
                par_R + (par_Rmax - par_R) * penalite_graph ** float(reg.expo)
            )
        elif isinstance(reg, RegSigmoid):
            radius = law_chromatic * (
                par_R
                + (par_Rmax - par_R)
                * (1 + np.exp(-reg.steepness * (penalite_graph - reg.center))) ** -1
            )
            par_model = reg
        else:
            assert_never(reg)

    logging.info("Computation of the penality map : DONE")

    loc_penality_time = time.time()

    logging.info(f"Time of the step : {loc_penality_time - loc_max_time:.2f}")

    # =============================================================================
    #  ROLLING PIN
    # =============================================================================

    logging.info("Rolling pin is rolling : LOADING")

    mask = (distance > 0) & (distance < 2.0 * par_R)

    loop = "y"
    count_iter = 0
    k_factor = []

    while loop == "y":
        mask = np.zeros(1)
        radius[0] = radius[0] / 1.5
        keep = [0]
        j = 0
        R_old = par_R

        while len(wave) - j > 3:
            par_R = float(radius[j])  # take the radius from the penality law
            mask = (distance[j, :] > 0) & (
                distance[j, :] < 2.0 * par_R
            )  # recompute the points closer than the diameter if Radius changed with the penality
            while np.sum(mask) == 0:
                par_R *= 1.5
                mask = (distance[j, :] > 0) & (
                    distance[j, :] < 2.0 * par_R
                )  # recompute the points closer than the diameter if Radius changed with the penality
            p1 = np.array([wave[j], flux[j]]).T  # vector of all the local maxima
            p2 = np.array(
                [wave[mask], flux[mask]]
            ).T  # vector of all the maxima in the diameter zone
            delta = p2 - p1  # delta x delta y
            c = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)  # euclidian distance
            h = np.sqrt(par_R**2 - 0.25 * c**2)
            cx = (
                p1[0] + 0.5 * delta[:, 0] - h / c * delta[:, 1]
            )  # x coordinate of the circles center
            cy = (
                p1[1] + 0.5 * delta[:, 1] + h / c * delta[:, 0]
            )  # y coordinates of the circles center

            cond1 = (cy - p1[1]) >= 0
            thetas = cond1 * (-1 * np.arccos((cx - p1[0]) / par_R) + np.pi) + (1 - 1 * cond1) * (
                -1 * np.arcsin((cy - p1[1]) / par_R) + np.pi
            )
            j2 = thetas.argmin()
            j = numero[mask][
                j2
            ]  # take the numero of the local maxima falling in the diameter zone
            keep.append(j)
        flux = flux[keep]  # we only keep the local maxima with the rolling pin condition
        wave = wave[keep]
        index = index[keep]
        if True:
            loop = "n"

    logging.info(" Rolling pin is rolling : DONE")

    loc_rolling_time = time.time()

    logging.info(f" Time of the step : {loc_rolling_time - loc_penality_time:.2f}")

    # =============================================================================
    # EDGE CUTTING
    # =============================================================================

    logging.info("Edge cutting : LOADING")

    count_cut = 0
    # TODO: remove this
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

    # TODO: do we need to?
    wave_backup = wave.copy()
    flux_backup = flux.copy()
    index_backup = index.copy()
    criterion = 1
    parameter = np.zeros(())  # TODO: initialize
    for j in range(5):
        diff_x = np.log10(
            np.min(np.vstack([abs(np.diff(wave)[1:]), abs(np.diff(wave)[0:-1])]), axis=0)
        )
        diff_diff_x = np.log10(abs(np.diff(np.diff(wave))) + 1e-5)

        diff_x = np.hstack([0, diff_x, 0])
        diff_diff_x = np.hstack([0, diff_diff_x, 0])
        if criterion == 1:
            parameter = diff_x - diff_diff_x
        elif criterion == 2:
            parameter = diff_x
        IQ = 2 * (np.nanpercentile(parameter, 50) - np.nanpercentile(parameter, 25))
        mask_out = parameter < (np.nanpercentile(parameter, 50) - 1.5 * IQ)
        if not sum(mask_out):
            criterion += 1
            if criterion == 2:
                continue
            elif criterion == 3:
                break
        mask_out_idx = np.arange(len(parameter))[mask_out]
        if len(mask_out_idx) > 1:
            cluster_idx = ras.clustering(mask_out_idx, 3, 1)
            unique = np.setdiff1d(mask_out_idx, np.hstack(cluster_idx))
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

    if len(wave) != len(wave_backup):
        #        plt.plot(grid, spectre, zorder=0)
        #        for j in np.setdiff1d(wave_backup,wave):
        #            plt.axvline(x=j,color='k',ls='-')
        #        Interpol = interp1d(wave, flux, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
        #        continuum = Interpol(grid)
        #        continuum = truncated(continuum)
        #        plt.plot(grid,continuum,ls=':',label='new continuum')
        #        Interpol = interp1d(wave_backup, flux_backup, kind = interpol, bounds_error = False, fill_value = 'extrapolate')
        #        continuum = Interpol(grid)
        #        continuum = truncated(continuum)
        #        plt.plot(grid,continuum,label = 'old continuum')
        #        plt.legend()
        #        plt.show()
        #        answer = sphinx('Do you accept the following grid rearangement ? (y/n)',rep=['y','n'])
        answer = "y"
        if answer == "n":
            wave = wave_backup.copy()
            flux = flux_backup.copy()

    logging.info(
        f"Number of points removed to build a more equidistant grid : {(len(wave_backup) - len(wave))}"
    )

    # =============================================================================
    # MANUAL ADDING/SUPRESSING
    # =============================================================================

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
        f"[END] RASSINE has finished to compute your continuum in {end - begin:.2f} seconds \n"
    )

    jump_point = 1  # make lighter figure for article

    if interpol == "cubic":
        continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(
            wave,
            flux,
            flux_denoised,
            grid,
            spectrei,
            continuum_to_produce=[interpol, "undenoised"],
        )
        conti = continuum3
    elif interpol == "linear":
        continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(
            wave,
            flux,
            flux_denoised,
            grid,
            spectrei,
            continuum_to_produce=[interpol, "undenoised"],
        )
        conti = continuum1
    else:
        raise NotImplementedError  # TODO: handle

    continuum1, continuum3, continuum1_denoised, continuum3_denoised = ras.make_continuum(
        wave,
        flux,
        flux_denoised,
        grid,
        spectrei,
        continuum_to_produce=[outputs_interpolation_saved, outputs_denoising_saved],
    )

    if (plot_end) | (save_last_plot):
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(2, 1, 1)
        plt.plot(
            grid[::jump_point],
            spectrei[::jump_point],
            label=f"spectrum (SNR={int(SNR_0):.0f})",
            color="g",
        )
        plt.plot(
            grid[::jump_point],
            spectre[::jump_point] * normalisation,
            label="spectrum reduced",
            color="b",
            alpha=0.3,
        )
        plt.scatter(wave, flux, color="k", label=f"anchor points ({int(len(wave))})", zorder=100)

    if (plot_end) | (save_last_plot):
        plt.plot(grid[::jump_point], conti[::jump_point], label="continuum", zorder=101, color="r")
        plt.xlabel("Wavelength", fontsize=14)
        plt.ylabel("Flux", fontsize=14)
        plt.legend(loc=4)
        plt.title("Final products of RASSINE", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        plt.tick_params(direction="in", top=True, which="both")
        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(grid[::jump_point], spectrei[::jump_point] / conti[::jump_point], color="k")
        plt.axhline(y=1, color="r", zorder=102)
        plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
        plt.ylabel("Flux normalised", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        plt.tick_params(direction="in", top=True, which="both")
        plt.subplots_adjust(left=0.07, right=0.96, hspace=0, top=0.95)
        if save_last_plot:
            plt.savefig(output_dir / f"{new_file}_output.png")
        # TODO: remove this
        plt.close()

    # =============================================================================
    # SAVE OF THE PARAMETERS
    # =============================================================================

    if (hole_left is not None) & (hole_left != -99.9):
        hole_left = ras.find_nearest1(grid, ras.doppler_r(hole_left, -30)[0])[1]
    if (hole_right is not None) & (hole_right != -99.9):
        hole_right = ras.find_nearest1(grid, ras.doppler_r(hole_right, 30)[0])[1]

    def print_parameters_according_to_paper():
        # TODO: move the name_parameters stuff inside here
        pass

    parameters = {
        "number_iteration": count_iter,
        "K_factors": k_factor,
        "axes_stretching": np.round(par_stretching, 1),
        "vicinity_local_max": par_vicinity,
        "smoothing_box": smoothing_length,
        "smoothing_kernel": smoothing_shape,
        "fwhm_ccf": np.round(par_fwhm / conversion_fwhm_sig, 2),
        "CCF_mask": CCF_mask,
        "RV_sys": RV_sys,
        "min_radius": np.round(R_old, 1),
        "max_radius": np.round(par_Rmax, 1),
        "model_penality_radius": par_model,
        "denoising_dist": denoising_dist,
        "number of cut": count_cut,
        "windows_penality": windows,
        "large_window_penality": big_windows,
        "number_points": len(grid),
        "number_anchors": len(wave),
        "SNR_5500": int(SNR_0),
        "mjd": mjd,
        "jdb": jdb,
        "wave_min": minx,
        "wave_max": maxx,
        "dwave": dgrid,
        "hole_left": hole_left,
        "hole_right": hole_right,
        "RV_shift": RV_shift,
        "berv": berv,
        "lamp_offset": lamp_offset,
        "acc_sec": acc_sec,
        "light_file": True,
        "speedup": speedup,
        "continuum_interpolated_saved": outputs_interpolation_saved,
        "continuum_denoised_saved": outputs_denoising_saved,
        "nb_spectra_stacked": nb_spectra_stacked,
        "arcfiles": arcfiles,
    }

    name_parameters = [
        "number_iteration",
        "K_factors",
        "par_stretching",
        "par_vicinity",
        "par_smoothing_box",
        "par_smoothing_kernel",
        "par_fwhm",
        "CCF_mask",
        "RV_sys",
        "par_R",
        "par_Rmax",
        "par_reg_nu",
        "denoising_dist",
        "count_cut_lim",
        "windows_penality",
        "large_window_penality",
        "number of points",
        "number of anchors",
        "SNR_5500",
        "mjd",
        "jdb",
        "wave_min",
        "wave_max",
        "dwave",
        "hole_left",
        "hole_right",
        "RV_shift",
        "berv",
        "lamp_offset",
        "acc_sec",
        "light_file",
        "speedup",
        "continuum_interpolated_saved",
        "continuum_denoised_saved",
        "nb_spectra_stacked",
        "arcfiles",
    ]

    if logging.getLogger().level <= 20:  # logging INFO
        print("\n------TABLE------- \n")
        for i, j in zip(name_parameters, parameters.keys()):
            print(i + " : " + str(parameters[j]))
        print("\n----------------- \n")

    if not Penalty:
        penalite_step = None
        penalite0 = None

    # conversion in fmt format

    flux_used = spectre * normalisation
    index = index.astype("int")

    # =============================================================================
    # SAVE THE OUTPUT
    # =============================================================================

    # TODO: mode "basic" with the stuff that used later in the pipeline
    #       mode "full" with everything

    # linear / undenoised
    # to discuss further

    if (outputs_interpolation_saved == "linear") & (outputs_denoising_saved == "undenoised"):
        basic = {
            "continuum_linear": continuum1,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "cubic") & (outputs_denoising_saved == "undenoised"):
        basic = {
            "continuum_cubic": continuum3,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "linear") & (outputs_denoising_saved == "denoised"):
        basic = {
            "continuum_linear": continuum1_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux_denoised,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "cubic") & (outputs_denoising_saved == "denoised"):
        basic = {
            "continuum_cubic": continuum3_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux_denoised,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "all") & (outputs_denoising_saved == "denoised"):
        basic = {
            "continuum_cubic": continuum3_denoised,
            "continuum_linear": continuum1_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux_denoised,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "all") & (outputs_denoising_saved == "undenoised"):
        basic = {
            "continuum_cubic": continuum3,
            "continuum_linear": continuum1,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "linear") & (outputs_denoising_saved == "all"):
        basic = {
            "continuum_linear": continuum1,
            "continuum_linear_denoised": continuum1_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_flux_denoised": flux_denoised,
            "anchor_index": index,
        }
    elif (outputs_interpolation_saved == "cubic") & (outputs_denoising_saved == "all"):
        basic = {
            "continuum_cubic": continuum3,
            "continuum_cubic_denoised": continuum3_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_flux_denoised": flux_denoised,
            "anchor_index": index,
        }
    else:
        basic = {
            "continuum_cubic": continuum3,
            "continuum_linear": continuum1,
            "continuum_cubic_denoised": continuum3_denoised,
            "continuum_linear_denoised": continuum1_denoised,
            "anchor_wave": wave,
            "anchor_flux": flux,
            "anchor_flux_denoised": flux_denoised,
            "anchor_index": index,
        }

    # we assume light_version=True here
    output = {
        "wave": grid,
        "flux": spectrei,
        "flux_used": flux_used,
        "output": basic,
        "parameters": parameters,
    }

    if spectrei_err is not None:
        output["flux_err"] = spectrei_err

    output["parameters"]["filename"] = "RASSINE_" + new_file + ".p"

    output_file = output_dir / f"RASSINE_{new_file}.p"
    ras.save_pickle(output_file, output)
    print(
        f"Output file saved under : {output_file} (SNR at 5500 : {output['parameters']['SNR_5500']:.0f})"
    )
