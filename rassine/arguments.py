import argparse
import os
import re
from typing import List, Literal, Tuple, Union

float_regex = r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?"


def par_stretching_type(s: str) -> Union[str, float]:
    """
    Parses a par_stretching argument

    Args:
       s: String argument to parse

    Returns:
        The parsed value, either a float or a string with format "auto_XXX" where XXX is a float
    """
    auto_regex = r"auto_-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?"
    if re.fullmatch(float_regex, s):
        return float(s)
    elif re.fullmatch(auto_regex, s):
        return s


def nonneg_int_or_auto(s: str) -> Union[Literal["auto"], int]:
    """
    Parses a string argument that contains either a non-negative integer or the 'auto' string

    Args:
       s: String argument to parse

    Returns:
       The parsed value
    """
    if s == "auto":
        return "auto"
    elif s.isdigit():
        return int(s)
    else:
        raise argparse.ArgumentTypeError(
            "'{}' is not a valid non-negative integer or 'auto'".format(s)
        )


def float_or_auto(s: str) -> Union[Literal["auto"], float]:
    """
    Parses a string argument that contains either a floating-point number or the 'auto' string

    Args:
        s: String argument to parse

    Returns:
        The parsed value
    """
    if s == "auto":
        return "auto"
    else:
        try:
            return float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "'{}' is not a valid floating-point number or 'auto'".format(s)
            )

    # types taken from rassine_config.py


def mask_telluric_type(s: str) -> List[Tuple[float, float]]:
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
    return pairs.parse(s)


def str2bool(s: str) -> bool:
    """
    Parses a string argument that represents a Boolean value

    Args:
        s: String argument to parse

    Returns:
        The parsed boolean value
    """
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got '{}' instead.".format(s))


class StoreExceptUnspecifiedAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        if nargs == 0:
            raise ValueError(
                "nargs for store actions must be != 0; if you "
                "have nothing to store, actions such as store "
                "true or store const may be more appropriate"
            )
        if const is not None and nargs != OPTIONAL:
            raise ValueError("nargs must be %r to supply const" % OPTIONAL)
        super(StoreExceptUnspecifiedAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values != "unspecified":
            setattr(namespace, self.dest, values)


class SpectrumAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        if nargs == 0:
            raise ValueError(
                "nargs for store actions must be != 0; if you "
                "have nothing to store, actions such as store "
                "true or store const may be more appropriate"
            )
        if const is not None and nargs != OPTIONAL:
            raise ValueError("nargs must be %r to supply const" % OPTIONAL)
        super(SpectrumAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        spectrum_name = values
        setattr(namespace, "output_dir", os.path.dirname(spectrum_name) + "/")


def argument_parser(cwd: str = "") -> argparse.ArgumentParser:
    """
    Creates an argument parser for RASSINE parameters

    Returns:
        An argument parser
    """
    parser = argparse.ArgumentParser(description="RASSINE tool")
    parameters = parser.add_argument_group(title="Algorithm parameters")

    parser.add_argument(
        "--spectrum_name",
        "-s",
        type=str,
        action=SpectrumAction,
        default=cwd + "/spectra_library/spectrum_cenB.csv",
        help="Full path of your spectrum pickle/csv file",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=cwd + "/output/",
        action=StoreExceptUnspecifiedAction,
        help="Directory where output files are written",
    )

    parser.add_argument(
        "--synthetic_spectrum",
        type=bool,
        default=False,
        help="True if working with a noisy-free synthetic spectra",
    )

    parser.add_argument(
        "--anchor_file",
        "-l",
        type=str,
        default="",
        help="Put a RASSINE output file that will fix the value of the 7 parameters to the same value than in the anchor file",
    )

    parser.add_argument("--column_wave", type=str, default="wave")
    parser.add_argument("--column_flux", type=str, default="flux")

    parser.add_argument(
        "--float_precision",
        type=str,
        default="float32",
        help="Float precision for the output products wavelength grid",
    )

    parameters.add_argument(
        "--par_stretching",
        "-p",
        type=par_stretching_type,
        default="auto_0.5",
        help="Stretch the x and y axes ratio ('auto_X' available)",
    )

    parser.add_argument(
        "--par_vicinity", type=int, default=7, help="Half-window to find a local maxima"
    )

    parameters.add_argument(
        "--par_smoothing_box",
        type=nonneg_int_or_auto,
        default=6,
        help="Half-window of the box used to smooth (1 => no smoothing, 'auto' available)",
    )

    # par_smoothing_box = 'auto' implies either 'erf', 'hat_exp'
    parser.add_argument(
        "--par_smoothing_kernel",
        type=str,
        default="savgol",
        choices=["rectangular", "gaussian", "savgol", "erf", "hat_exp"],
        help="If par_smoothing_box is an integer, valid options are rectangular, gaussian, savgol. If par_smoothing_box is auto, valid options are erf and hat_exp",
    )

    parameters.add_argument(
        "--par_fwhm",
        type=float_or_auto,
        default="auto",
        help="FWHM of the CCF in km/s ('auto' available)",
    )

    parser.add_argument(
        "--CCF_mask", type=str, default="master", help="Only needed if par_fwhm is in 'auto'"
    )
    parser.add_argument(
        "--RV_sys",
        type=float,
        default=0.0,
        help="RV systemic in kms, only needed if par_fwhm is in 'auto' and CCF different of 'master'",
    )

    parser.add_argument(
        "--mask_telluric",
        type=mask_telluric_type,
        default="[[6275,6330],[6470,6577],[6866,8000]]",
        help="A list of left and right borders to eliminate from the mask of the CCF only if CCF = 'master' and par_fwhm = 'auto'",
    )

    parameters.add_argument(
        "--par_R",
        "-r",
        type=float_or_auto,
        default="auto",
        help="Minimum radius of the rolling pin in angstrom ('auto' available)",
    )
    parameters.add_argument(
        "--par_Rmax",
        "-R",
        type=float_or_auto,
        default="auto",
        help="Maximum radius of the rolling pin in angstrom ('auto' available)",
    )
    parameters.add_argument(
        "--par_reg_nu",
        type=str,
        default="poly_1.0",
        help="Penality-radius law, either poly_d (d the degree of the polynome x**d) or or sigmoid_c_s where c is the center and s the steepness",
    )  # TODO why the comma

    parser.add_argument(
        "--denoising_dist",
        type=int,
        default=5,
        help="Half window of the area used to average the number of point around the local max for the continuum",
    )
    parser.add_argument(
        "--count_cut_lim",
        type=int,
        default=3,
        help="Number of border cut in automatic mode (put at least 3 if Automatic mode)",
    )
    parser.add_argument(
        "--count_out_lim",
        type=int,
        default=1,
        help="Number of outliers clipping in automatic mode (put at least 1 if Automatic mode)",
    )
    parser.add_argument(
        "--interpol",
        type=str,
        default="cubic",
        help="Define the interpolation for the continuum displayed in the subproducts; note that at the end a cubic and linear interpolation are saved in 'output' regardless this value",
    )
    parser.add_argument(
        "--feedback",
        "-a",
        type=str2bool,
        default=True,
        action=StoreExceptUnspecifiedAction,
        help="Run the code without graphical feedback and interactions with the sphinx (only wishable if lot of spectra)",
    )
    parser.add_argument("--only_print_end", "-P", type=str2bool, default=False, help="")
    parser.add_argument("--plot_end", "-e", type=str2bool, default=True, help="")
    parser.add_argument("--save_last_plot", "-S", type=str2bool, default=False, help="")
    parser.add_argument(
        "--outputs_interpolation_saved",
        type=str,
        default="linear",
        choices=["linear", "cubic", "all"],
        help="To only save a specific continuum (output files are lighter), either 'linear','cubic' or 'all'",
    )
    parser.add_argument(
        "--outputs_denoising_saved",
        type=str,
        default="undenoised",
        choices=["denoised", "undenoised", "all"],
        help="To only save a specific continuum (output files are lighter), either 'denoised','undenoised' or 'all'",
    )
    parser.add_argument(
        "--light_version", type=str2bool, default=True, help="To save only the vital output"
    )
    parser.add_argument(
        "--speedup",
        type=int,
        default=1,
        help="To improve the speed of the rolling processes (not yet fully tested)",
    )
    return parser
