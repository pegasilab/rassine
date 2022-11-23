"""
Main file for the RASSINE script

"""

from __future__ import annotations

import argparse
import logging
import textwrap
import typing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union

import configpile as cp
import matplotlib
import matplotlib.pylab as plt
import rich
import rich.pretty
import tybles as tb
from matplotlib.ticker import MultipleLocator
from typeguard import check_type
from typing_extensions import Annotated

from ..lib.data import LoggingLevel, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..stacking.data import MasterPickle, StackedBasicRow, StackedPickle
from .data import ExtraPlotData, RassineParameters, RassinePickle
from .parsing import (
    Auto,
    Reg,
    Stretching,
    auto_float_parser,
    auto_int_parser,
    mask_telluric_parser,
    reg_parser,
    stretching_parser,
)
from .process import rassine_process


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Rolling Alpha Shape for a Spectral Improved Normalisation Estimator (RASSINE)

    ::
    
             ^                  .-=-.          .-==-.
            {}      __        .' O o '.       /   ^  )
           { }    .' O'.     / o .-. O \\     /  .--`\\
           { }   / .-. o\\   /O  /   \\  o\\   /O /    ^  (RASSSSSSINE)
            \\ `-` /   \\ O`-'o  /     \\  O`-`o /
        jgs  `-.-`     '.____.'       `.____.'

    Authors: Michael Cretignier, Jeremie Francfort and Denis Rosset
    """

    #
    # Common information
    #

    prog_ = Path(__file__).stem
    ini_strict_sections_ = [Path(__file__).stem]
    ini_relaxed_sections_ = [Path(__file__).stem.split("_")[0]]
    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], cp.Param.config(env_var_name="RASSINE_CONFIG")]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, cp.Param.root_path(env_var_name="RASSINE_ROOT")]

    #: Pickle protocol version to use
    pickle_protocol: Annotated[
        PickleProtocol, cp.Param.store(PickleProtocol.parser(), default_value="3")
    ]

    #: Logging level to use
    logging_level: Annotated[
        LoggingLevel,
        cp.Param.store(
            LoggingLevel.parser(), default_value="WARNING", env_var_name="RASSINE_LOGGING_LEVEL"
        ),
    ]

    #
    # Task specific information
    #

    #: Path of the spectrum pickle
    #:
    #: Either a path to a spectrum pickle needs to be provided, or one needs to provide
    #: the --input-table, --input-folder and inputs (positional) arguments
    #:
    #: Note that input_spectrum is a path relative to "root" and "input_folder"
    input_spectrum: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-s", default_value=""
        ),
    ]

    #: Table of stacked spectra from which to read the individual file names
    input_table: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-I", default_value=""
        ),
    ]

    #: Relative path to the folder containing the stacked spectra
    input_folder: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-i", default_value=""
        ),
    ]

    #: Indices of spectrum to process
    #:
    #: If provided, then input_table and input_folder must be provided
    #: If not provided, then input_spectrum must be provided
    input_indices: Annotated[
        Sequence[int],
        cp.Param.append1(
            cp.parsers.int_parser,
            positional=cp.Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    def validate_inputs(self) -> Optional[cp.Err]:
        errors: List[Optional[cp.Err]] = []
        if self.input_indices:
            errors.append(
                cp.Err.check(
                    self.input_table is not None,
                    "If input_indices are provided, input_table must be provided",
                )
            )
            errors.append(
                cp.Err.check(
                    self.input_folder is not None,
                    "If input_indices are provided, input_folder must be provided",
                )
            )
            errors.append(
                cp.Err.check(
                    self.input_spectrum is None,
                    "If input_indices are provided, then input_spectrum must not be provided",
                )
            )
        else:
            errors.append(
                cp.Err.check(
                    self.input_spectrum is not None,
                    "At least one of input_indices or input_spectrum must be provided",
                )
            )
        if self.input_spectrum is not None:
            errors.append(
                cp.Err.check(
                    self.input_table is None,
                    "If input_spectrum is provided, then input_table must not be provided",
                )
            )
        return cp.Err.collect(*errors)

    #: Directory where output files are written
    output_folder: Annotated[
        Path,
        cp.Param.store(cp.parsers.path_parser, short_flag_name="-o"),
    ]

    #: Pattern to use for the RASSINE output filename
    #:
    #: The "{}" part will be replaced by the input filename stem
    output_pattern: Annotated[
        str, cp.Param.store(cp.parsers.str_parser, default_value="RASSINE_{}.p")
    ]

    #: Put a RASSINE output file that will fix the value of the 7 parameters to the same value than in the anchor file
    input_anchor_pickle: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-l", default_value=""
        ),
    ]

    def validate_input_anchor_pickle(self) -> Optional[cp.Err]:
        return cp.Err.check(
            self.input_anchor_pickle is None or (self.root / self.input_anchor_pickle).exists(),
            "Anchor pickle file, if provided, must exist",
        )

    #: Optional INI output anchor file that can be used as an input configuration file
    #:
    #: Only valid if input_spectrum is used and not the multi-spectrum reduction
    output_anchor_ini: Annotated[
        Optional[Path],
        cp.Param.store(cp.parsers.path_parser.empty_means_none(), default_value=""),
    ]

    def validate_output_anchor_ini(self) -> Optional[cp.Err]:
        if self.output_anchor_ini is not None:
            return cp.Err.check(
                self.input_spectrum is not None,
                "If output_anchor_ini is provided, then one must provide input_spectrum as well",
            )
        return None

    #: Where to put the output plot, if empty no plot is produced
    #:
    #: The plot name will be derived from the spectrum name, with a "_output.png" suffix
    output_plot_folder: Annotated[
        Optional[Path],
        cp.Param.store(cp.parsers.path_parser.empty_means_none(), default_value=""),
    ]

    #: Pattern to use for the RASSINE output filename
    #:
    #: The "{}" part will be replaced by the input filename stem
    output_plot_pattern: Annotated[
        str,
        cp.Param.store(cp.parsers.str_parser.empty_means_none(), default_value="{}_output.png"),
    ]

    #: Scaling of the flux axis compared to the wavelength axis. 
    #: The format of the automatic mode is'auto x' with x a 1 decimal positive float number. 
    #: x = 0.0 means high tension, whereas x = 1.0 mean low tension.
    #:
    #: PARAMETER 1
    par_stretching: Annotated[
        Stretching,
        cp.Param.store(stretching_parser, default_value="auto_0.5", short_flag_name="-p"),
    ]

    def validate_par_stretching(self) -> Optional[cp.Err]:
        if isinstance(self.par_stretching, Auto):
            if not (0 <= self.par_stretching.ratio <= 1):
                return cp.Err.make("the par_stretching auto value must be between 0 and 1")
        else:
            if self.par_stretching < 0.0:
                return cp.Err.make("the par_stretching fixed value cannot be negative")
        return None

    #: half-window size to find a local maxima
    par_vicinity: Annotated[int, cp.Param.store(cp.parsers.int_parser, default_value="7")]

    #: half-window of the box used to smooth (1 => no smoothing, 'auto' available)
    #:
    #: PARAMETER 2
    par_smoothing_box: Annotated[
        Union[Literal["auto"], int], cp.Param.store(auto_int_parser, default_value="6")
    ]
    #: To use the automatic mode which apply a Fourier filtering use 'erf ' or 'hat exp' kernel and 'auto' in par smoothing box. 
    #: Else, use 'rectangular', 'gaussian', 'savgol'. 
    #: Developers advise the 'savgol' kernel except if the user is dealing with spectra spanning low and high SNR range.
    par_smoothing_kernel: Annotated[
        Literal["rectangular", "gaussian", "savgol", "erf", "hat_exp"],
        cp.Param.store(
            cp.Parser.from_choices(["rectangular", "gaussian", "savgol", "erf", "hat_exp"]),
            default_value="savgol",
        ),
    ]

    def validate_smoothing(self) -> Optional[cp.Err]:
        if self.par_smoothing_box == "auto":
            if self.par_smoothing_kernel not in ["erf", "hat_exp"]:
                return cp.Err.make(
                    "If par_smoothing_box is auto, par_smoothing_kernel should be erf or hat_exp"
                )
        else:
            if self.par_smoothing_kernel not in ["rectangular", "gaussian", "savgol"]:
                return cp.Err.make(
                    "If par_smoothing_box is an integer, par_smoothing_kernel should be rectangular, gaussian or savgol"
                )
        return None

    #: FWHM of the CCF of the spectrum in km/s. The user can let 'auto' to let RASSINE determine this value by itself.
    #:
    #: PARAMETER 3
    par_fwhm: Annotated[
        Union[Literal["auto"], float], cp.Param.store(auto_float_parser, default_value="auto")
    ]

    #: CCF mask used to determine the FWHM. RASSINE construct its own mask by default. 
    #: The user can specify its own mask which should be placed in the CCF MASK directory. 
    #: Only needed if par_fwhm is in 'auto'
    CCF_mask: Annotated[str, cp.Param.store(cp.parsers.str_parser, default_value="master")]

    #: Minimum radius of the rolling pin in angstrom and in the extreme blue ('auto' available)
    #:
    #: PARAMETER 4
    par_R: Annotated[
        Union[Literal["auto"], float],
        cp.Param.store(auto_float_parser, default_value="auto", short_flag_name="-r"),
    ]

    #: Maximum radius of the rolling pin in angstrom in the extreme blue ('auto' available)
    #:
    #: PARAMETER 5
    par_Rmax: Annotated[
        Union[Literal["auto"], float],
        cp.Param.store(auto_float_parser, default_value="auto", short_flag_name="-R"),
    ]

    #: Penality-radius law
    #:
    #: poly_d (d the degree of the polynome x**d)
    #: or sigmoid_c_s where c is the center and s the steepness
    #:
    #: PARAMETER 6
    par_reg_nu: Annotated[Reg, cp.Param.store(reg_parser, default_value="poly_1.0")]

    #: a list of left and right borders to eliminate from the mask of the CCF
    #: only if CCF = 'master' and par_fwhm = 'auto'
    mask_telluric: Annotated[
        Sequence[Tuple[float, float]],
        cp.Param.store(
            mask_telluric_parser, default_value="[[6275,6330],[6470,6577],[6866,8000]]"
        ),
    ]

    #: True if working with a noisy-free synthetic spectra in order to inject a small noise for numerical stability
    synthetic_spectrum: Annotated[
        bool, cp.Param.store(cp.parsers.bool_parser, default_value="false")
    ]

    #: Define the interpolation for the continuum displayed in the subproducts
    #:
    #: note that at the end a cubic and linear interpolation are saved in 'output' regardless this value
    interpolation: Annotated[
        Literal["cubic", "linear"],
        cp.Param.store(cp.Parser.from_choices(["cubic", "literal"]), default_value="cubic"),
    ]

    #: Half window of the area used to average the number of point around the local max for the continuum
    denoising_dist: Annotated[int, cp.Param.store(auto_int_parser, default_value="5")]

    #: Number of border cut in automatic mode (put at least 3 if Automatic mode)
    count_cut_lim: Annotated[int, cp.Param.store(auto_int_parser, default_value="3")]

    #: Number of outliers clipping in automatic mode (put at least 1 if Automatic mode)
    count_out_lim: Annotated[int, cp.Param.store(auto_int_parser, default_value="1")]

    #: Random seed, derived from spectrum name if empty
    random_seed: Annotated[
        Optional[int], cp.Param.store(cp.parsers.int_parser.empty_means_none(), default_value="")
    ]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


def print_parameters_according_to_paper(parameters: RassineParameters):
    translation: Mapping[str, str] = {
        "filename": "filename",
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
        "rv_mean_jdb": "rv_mean_jdb",
        "rv_dace": "rv_dace",
        "rv_dace_std": "rv_dace_std",
    }
    rich.print("[bold red]Output parameters[/bold red]")
    new_dict = {translation[key]: value for key, value in parameters.items()}
    rich.pretty.pprint(new_dict)


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
    plt.close()


def update_using_anchor_file(task: Task, anchor_file: Path) -> Task:
    data = open_pickle(anchor_file)
    return replace(
        task,
        par_stretching=data["parameters"]["axes_stretching"],
        par_vicinity=data["parameters"]["vicinity_local_max"],
        par_smoothing_box=data["parameters"]["smoothing_box"],
        par_smoothing_kernel=data["parameters"]["smoothing_kernel"],
        par_fwhm=data["parameters"]["fwhm_ccf"],
        par_R=data["parameters"]["min_radius"],
        par_Rmax=data["parameters"]["max_radius"],
        par_reg_nu=data["parameters"]["model_penality_radius"],
        count_cut_lim=data["parameters"]["number_of_cut"],
    )


def make_ini_contents_from_parameters(
    params: RassineParameters, input_file: Optional[Path] = None
) -> str:
    return textwrap.dedent(
        f"""
        # Anchor file created on {datetime.now().isoformat()} from {input_file}
        [rassine]
        par-stretching={params["axes_stretching"]}
        par-vicinity={params["vicinity_local_max"]}
        par-smoothing-box={params["smoothing_box"]}
        par-smoothing-kernel={params["smoothing_kernel"]}
        par-fwhm={params["fwhm_ccf"]}
        par-R={params["min_radius"]}
        par-Rmax={params["max_radius"]}
        par-reg-nu={params["model_penality_radius"]}
        count-cut-lim={params["number_of_cut"]}
    """
    ).strip()


def run(t: Task):
    t.pickle_protocol.set()
    t.logging_level.set()
    if logging.getLogger().level <= 20:  # logging INFO
        rich.pretty.pprint(t)

    if t.input_anchor_pickle is not None:
        update_using_anchor_file(t, t.root / t.input_anchor_pickle)

    (t.root / t.output_folder).mkdir(parents=True, exist_ok=True)

    if t.output_anchor_ini is not None:
        (t.root / t.output_anchor_ini).parent.mkdir(parents=True, exist_ok=True)

    if t.output_plot_folder is not None:
        (t.root / t.output_plot_folder).mkdir(parents=True, exist_ok=True)

    @dataclass(frozen=True)
    class Step:
        stem: str
        row: Optional[StackedBasicRow]
        input_file: Path
        output_file: Path
        output_plot_file: Optional[Path]

    def output_plot_file_for_stem(stem: str) -> Optional[Path]:
        if t.output_plot_folder is not None:
            return t.root / t.output_plot_folder / (t.output_plot_pattern.replace("{}", stem))
        else:
            return None

    if t.input_indices:
        # do batch processing
        assert t.input_table is not None
        assert t.input_folder is not None
        logging.debug(f"Reading {t.root/t.input_table}")
        tyble = StackedBasicRow.schema().read_csv(t.root / t.input_table, return_type="Tyble")
        stems = [tyble[i].name for i in t.input_indices]
        steps: List[Step] = []
        for i in t.input_indices:
            row = tyble[i]
            stem = row.name
            steps.append(
                Step(
                    stem=stem,
                    row=tyble[i],
                    input_file=t.root / t.input_folder / (stem + ".p"),
                    output_file=t.root / t.output_folder / (t.output_pattern.replace("{}", stem)),
                    output_plot_file=output_plot_file_for_stem(stem),
                )
            )
    else:
        if t.input_folder is None:
            abs_input_folder = t.root
        else:
            abs_input_folder = t.root / t.input_folder
        assert t.input_spectrum is not None
        stem = t.input_spectrum.stem
        steps = [
            Step(
                stem,
                None,
                abs_input_folder / t.input_spectrum,
                t.root / t.output_folder / (t.output_pattern.replace("{}", stem)),
                output_plot_file_for_stem(stem),
            )
        ]

    if t.output_plot_folder is not None:
        plt.close("all")

    for step in steps:
        # compute random seed from input filename if not provided, so the randomization
        # is deterministic between runs on the same files
        if t.random_seed is None:
            random_seed = hash(step.input_file.stem)
        else:
            random_seed = t.random_seed

        data: Union[MasterPickle, StackedPickle] = open_pickle(step.input_file)

        data_id: List[str] = []
        try:
            check_type("data", data, MasterPickle)
            data_id.append("MasterPickle")
        except TypeError as e:
            pass

        try:
            check_type("data", data, StackedPickle)
            data_id.append("StackedPickle")
        except TypeError as e:
            pass

        logging.info(f"Read input pickle {step.input_file}, types identified: {data_id}")

        output, extra_plot_data = rassine_process(
            output_filename=step.output_file.name,
            row=step.row,
            data=data,
            synthetic_spectrum=t.synthetic_spectrum,
            random_seed=random_seed,
            par_stretching=t.par_stretching,
            par_vicinity=t.par_vicinity,
            par_smoothing_box=t.par_smoothing_box,
            par_smoothing_kernel=t.par_smoothing_kernel,
            par_fwhm_=t.par_fwhm,
            CCF_mask=t.CCF_mask,
            mask_telluric=t.mask_telluric,
            par_R_=t.par_R,
            par_Rmax_=t.par_Rmax,
            par_reg_nu=t.par_reg_nu,
            denoising_dist=t.denoising_dist,
            count_cut_lim=t.count_cut_lim,
            count_out_lim=t.count_out_lim,
            interpol=t.interpolation,
        )
        params = output["parameters"]
        save_pickle(step.output_file, output)
        logging.info(
            f"Output file saved under : {step.output_file} (SNR at 5500 : {params['SNR_5500']:.0f})"
        )

        if logging.getLogger().level <= 20:  # logging INFO
            print_parameters_according_to_paper(output["parameters"])

        if t.output_anchor_ini is not None:
            with open(t.root / t.output_anchor_ini, "wt") as f:
                f.write(make_ini_contents_from_parameters(output["parameters"], step.input_file))

        if step.output_plot_file is not None:
            plot_output(output, extra_plot_data, step.output_plot_file)


def cli() -> None:
    run(Task.from_command_line_())
