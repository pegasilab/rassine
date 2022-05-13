from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import configpile as cp
from typing_extensions import Annotated, TypeAlias

from ...io import open_pickle
from ...util import assert_never


@dataclass(frozen=True)
class RegPoly:
    """Penality-radius law, polynomial mapping, see Eq. (2) of RASSINE paper"""

    def name(self) -> str:
        return "poly"

    string: str
    expo: float


@dataclass(frozen=True)
class RegSigmoid:
    """Penality-radius law, logistic model"""

    def name(self) -> str:
        return "sigmoid"

    string: str
    center: float
    steepness: float


Reg: TypeAlias = Union[RegPoly, RegSigmoid]


@dataclass(frozen=True)
class Auto:
    ratio: float


Stretching: TypeAlias = Union[float, Auto]


def stretching_to_string(value: Stretching) -> str:
    if isinstance(value, Auto):
        return f"auto_{value.ratio}"
    else:
        return str(value)


def _stretching_parse(arg: str) -> Stretching:
    try:
        return float(arg)
    except:
        pass
    assert arg.startswith("auto_")
    return Auto(float(arg[5:]))


def _reg_parse(arg: str) -> Reg:
    parts = arg.split("_")
    if parts[0] == "poly":
        assert len(parts) == 2
        return RegPoly(string=arg, expo=float(parts[1]))
    elif parts[0] == "sigmoid":
        assert len(parts) == 3
        return RegSigmoid(string=arg, center=float(parts[1]), steepness=float(parts[2]))
    else:
        raise ValueError("Invalid reg")


reg_parser: cp.Parser[Reg] = cp.Parser.from_function_that_raises(_reg_parse)


stretching_parser: cp.Parser[Stretching] = cp.Parser.from_function_that_raises(_stretching_parse)


def _auto_float_parse(arg: str) -> Union[float, Literal["auto"]]:
    if arg.strip().lower() == "auto":
        return "auto"
    else:
        return float(arg)


auto_float_parser: cp.Parser[Union[float, Literal["auto"]]] = cp.Parser.from_function_that_raises(
    _auto_float_parse
)


def _auto_int_parse(arg: str) -> Union[int, Literal["auto"]]:
    if arg.strip().lower() == "auto":
        return "auto"
    else:
        return int(arg)


auto_int_parser: cp.Parser[Union[int, Literal["auto"]]] = cp.Parser.from_function_that_raises(
    _auto_int_parse
)


def _parse_mask_telluric(s: str) -> Sequence[Tuple[float, float]]:
    """
    Parses a list of telluric to mask

    Args:
        s: String argument to parse

    Returns:
        The parsed value
    """
    import parsy
    from parsy import seq, string

    # parse a number
    number = parsy.regex(r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?").map(float)
    pair = (string("[") >> seq(number << string(","), number) << string("]")).map(
        lambda s: (s[0], s[1])
    )
    pairs = string("[") >> pair.sep_by(string(",")) << string("]")
    res: Sequence[Tuple[float, float]] = pairs.parse(s)
    return res


mask_telluric_parser = cp.Parser.from_function_that_raises(_parse_mask_telluric)


@dataclass(frozen=True)
class Config(cp.Config):
    """
    =====================================================================================
    Rolling Alpha Shape for a Spectral Improved Normalisation Estimator (RASSINE)
    =====================================================================================  
         ^                  .-=-.          .-==-.
        {}      __        .' O o '.       /   ^  )
       { }    .' O'.     / o .-. O \\     /  .--`\\
       { }   / .-. o\\   /O  /   \\  o\\   /O /    ^  (RASSSSSSINE)
        \\ `-` /   \\ O`-'o  /     \\  O`-`o /
    jgs  `-.-`     '.____.'       `.____.'

    Authors: Michael Cretignier, Jeremie Francfort and Denis Rosset
    """

    #: Full path of the spectrum pickle
    spectrum_name: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-s")]

    #: Directory where output files are written
    #:
    #: If "unspecified" is given, taken from the spectrum_name folder
    output_dir: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-o", default_value=""
        ),
    ]

    #: Put a RASSINE output file that will fix the value of the 7 parameters to the same value than in the anchor file
    input_anchor_pickle: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-l", default_value=""
        ),
    ]

    # #: Optional INI output anchor file that can be used as an input configuration file
    # output_anchor_config: Annotated[
    #     Optional[Path],
    #     cp.Param.store(
    #         cp.parsers.path_parser.empty_means_none(), short_flag_name="-l", default_value=""
    #     ),
    # ]

    #: True if working with a noisy-free synthetic spectra
    synthetic_spectrum: Annotated[
        bool, cp.Param.store(cp.parsers.bool_parser, default_value="false")
    ]

    #: stretch the x and y axes ratio ('auto' available)
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

    #: half-window to find a local maxima
    par_vicinity: Annotated[int, cp.Param.store(cp.parsers.int_parser, default_value="7")]

    #: half-window of the box used to smooth (1 => no smoothing, 'auto' available)
    #:
    #: PARAMETER 2
    par_smoothing_box: Annotated[
        Union[Literal["auto"], int], cp.Param.store(auto_int_parser, default_value="6")
    ]

    #: 'rectangular','gaussian','savgol' if a value is specified in smoothing_kernel
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

    #: FWHM of the CCF in km/s ('auto' available)
    #:
    #: PARAMETER 3
    par_fwhm: Annotated[
        Union[Literal["auto"], float], cp.Param.store(auto_float_parser, default_value="auto")
    ]

    #: only needed if par_fwhm is in 'auto'
    CCF_mask: Annotated[str, cp.Param.store(cp.parsers.str_parser, default_value="master")]

    #: Minimum radius of the rolling pin in angstrom ('auto' available)
    #:
    #: PARAMETER 4
    par_R: Annotated[
        Union[Literal["auto"], float],
        cp.Param.store(auto_float_parser, default_value="auto", short_flag_name="-r"),
    ]

    #: Maximum radius of the rolling pin in angstrom ('auto' available)
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

    #: Define the interpolation for the continuum displayed in the subproducts
    #:
    #: note that at the end a cubic and linear interpolation are saved in 'output' regardless this value
    interpolation: Annotated[
        Literal["cubic", "linear"],
        cp.Param.store(cp.Parser.from_choices(["cubic", "literal"]), default_value="cubic"),
    ]

    #: Display the final product in the graphic
    plot_end: Annotated[
        bool, cp.Param.store(cp.parsers.bool_parser, default_value="true", short_flag_name="-e")
    ]

    #: Save the last graphical output (final output)
    save_last_plot: Annotated[
        bool, cp.Param.store(cp.parsers.bool_parser, default_value="false", short_flag_name="-S")
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


def update_using_anchor_file(config: Config, anchor_file: Path) -> Config:
    # TODO: update this
    data = open_pickle(anchor_file)
    return replace(
        config,
        par_stretching=data["parameters"]["axes_stretching"],
        par_vicinity=data["parameters"]["vicinity_local_max"],
        par_smoothing_box=data["parameters"]["smoothing_box"],
        par_smoothing_kernel=data["parameters"]["smoothing_kernel"],
        par_fwhm=data["parameters"]["fwhm_ccf"],
        par_R=data["parameters"]["min_radius"],
        par_Rmax=data["parameters"]["max_radius"],
        par_reg_nu=data["parameters"]["model_penality_radius"],
        count_cut_lim=data["parameters"]["number of cut"],
    )
