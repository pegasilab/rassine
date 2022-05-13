from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator

from ...io import open_pickle, save_pickle
from .config import Config, update_using_anchor_file
from .formats import ExtraPlotData, RassineParameters, RassinePickle
from .process import rassine_process

logging.getLogger().setLevel("INFO")

from typing import Mapping, Optional, Union

from ..stacking_master_spectrum import MasterPickle
from ..stacking_stack import StackedPickle


def print_parameters_according_to_paper(parameters: RassineParameters):
    translation: Mapping[str, str] = {
        "number_iteration": "number_iteration",
        "K_factors": "K_factors",
        "axes_stretching": "par_stretching",
        "vicinity_local_max": "par_vicinity",
        "smoothing_box": "par_smoothing_box",
        "smoothing_kernel": "par_smoothing_kernel",
        "fwhm_ccf": "par_fwhm",
        "CCF_mask": "CCF_mask",
        "RV_sys": "RV_sys",
        "min_radius": "par_R",
        "max_radius": "par_Rmax",
        "model_penality_radius": "par_reg_nu",
        "denoising_dist": "denoising_dist",
        "number_of_cut": "count_cut_lim",
        "windows_penality": "windows_penality",
        "large_window_penality": "large_window_penality",
        "number_points": "number of points",
        "number_anchors": "number of anchors",
        "SNR_5500": "SNR_5500",
        "mjd": "mjd",
        "jdb": "jdb",
        "wave_min": "wave_min",
        "wave_max": "wave_max",
        "dwave": "dwave",
        "hole_left": "hole_left",
        "hole_right": "hole_right",
        "RV_shift": "RV_shift",
        "berv": "berv",
        "lamp_offset": "lamp_offset",
        "acc_sec": "acc_sec",
        "light_file": "light_file",
        "speedup": "speedup",
        "continuum_interpolated_saved": "continuum_interpolated_saved",
        "continuum_denoised_saved": "continuum_denoised_saved",
        "nb_spectra_stacked": "nb_spectra_stacked",
        "arcfiles": "arcfiles",
    }
    print("\n------TABLE------- \n")
    for key, display in translation.items():
        print(f"{display} : {parameters[key]}")
    print("\n----------------- \n")


def plot_output(
    output: RassinePickle,
    extra: ExtraPlotData,
    output_file: Optional[Path] = None,
    jump_point: int = 1,
):
    """Plots RASSINE result

    Args:
        jump_point: Put > 1 to make lighter plot for article
    """
    grid = output["wave"]
    spectrei = output["flux"]
    wave = output["output"]["anchor_wave"]
    flux = output["output"]["anchor_flux"]
    normalisation = extra.normalisation
    spectre = extra.spectre
    conti = extra.conti
    SNR_0 = output["parameters"]["SNR_5500"]  # now rounded to integer
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
    if output_file is not None:
        plt.savefig(output_file)
    # TODO: remove this
    plt.close()


def cli():
    cfg: Config = Config.from_command_line_()
    if cfg.random_seed is None:
        random_seed = hash(cfg.spectrum_name.stem)
    else:
        random_seed = cfg.random_seed
    spectrum_name = cfg.spectrum_name

    if cfg.output_dir is None:
        output_dir = spectrum_name.parent
    else:
        output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.input_anchor_pickle is not None:
        assert cfg.input_anchor_pickle.exists(), "Anchor file, if provided, must exist"
        update_using_anchor_file(cfg, cfg.input_anchor_pickle)

    # load the pickle dictionary
    data: Union[MasterPickle, StackedPickle] = open_pickle(spectrum_name, MasterPickle)
    output_filename = f"RASSINE_{spectrum_name.stem}.p"

    output, extra_plot_data = rassine_process(
        output_filename=output_filename,
        data=data,
        synthetic_spectrum=cfg.synthetic_spectrum,
        random_seed=random_seed,
        par_stretching=cfg.par_stretching,
        par_vicinity=cfg.par_vicinity,
        par_smoothing_box=cfg.par_smoothing_box,
        par_smoothing_kernel=cfg.par_smoothing_kernel,
        par_fwhm_=cfg.par_fwhm,
        CCF_mask=cfg.CCF_mask,
        mask_telluric=cfg.mask_telluric,
        par_R_=cfg.par_R,
        par_Rmax_=cfg.par_Rmax,
        par_reg_nu=cfg.par_reg_nu,
        denoising_dist=cfg.denoising_dist,
        count_cut_lim=cfg.count_cut_lim,
        count_out_lim=cfg.count_out_lim,
        interpol=cfg.interpolation,
    )
    if logging.getLogger().level <= 20:  # logging INFO
        print_parameters_according_to_paper(output["parameters"])

    output_file = output_dir / output_filename
    save_pickle(output_file, output)
    print(
        f"Output file saved under : {output_file} (SNR at 5500 : {output['parameters']['SNR_5500']:.0f})"
    )

    if cfg.plot_end or cfg.save_last_plot:
        matplotlib.use("Qt5Agg", force=True)
        plt.close("all")

        if cfg.save_last_plot:
            plot_output_file: Optional[Path] = output_dir / f"{spectrum_name.stem}_output.png"
        else:
            plot_output_file = None
        plot_output(output, extra_plot_data, plot_output_file)
