import re
from dataclasses import dataclass
from enum import auto
from typing import Literal, Sequence, Tuple, Union

import configpile as cp
import configpile.parsers as cpp
import parsy as ps
from typing_extensions import Annotated

from .paths import RelPath

#: Regular expression that parses a floating point number
float_regex = r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?"


@dataclass(frozen=True)
class Auto:
    factor: float  #: factor between 0 and 1, see Fig B.4. of https://arxiv.org/pdf/2006.13098.pdf


class ParStretchingParser(cpp.Parser[Union[Auto, float]]):
    """
    Parser for the par_stretching argument

    Parses either:

    - a floating-point value, which is returned as it is
    - a floating-point value, prefixed by ``auto_``, which is returned wrapped in a
      :class:`.Auto` instance.
    """

    def parse(self, s: str) -> cp.Res[Union[Auto, float]]:
        # Regular expression that parses a floating point number prefixed with auto_
        auto_regex = r"auto_-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?"
        if re.fullmatch(float_regex, s):
            return float(s)
        elif re.fullmatch(auto_regex, s):
            return Auto(float(s.split("_")[1]))
        else:
            return cp.Err.make(f"Could not parse {s} as a par_stretching value")


class NonNegIntOrAutoParser(cpp.Parser[Union[Literal["auto"], int]]):
    """
    Parser for a non-negative integer or a "auto" string literal
    """

    def parse(self, arg: str) -> cp.Res[Union[Literal["auto"], int]]:
        arg = arg.strip().lower()
        if arg == "auto":
            return "auto"
        else:
            try:
                res = int(arg)
                if res < 0:
                    return cp.Err.make(f"Expected non-negative integer, got {res}")
                return res
            except ValueError as e:
                return cp.Err.make(str(e))


class FloatOrAutoParser(cpp.Parser[Union[Literal["auto"], float]]):
    """
    Parser for a floating-point number or a "auto" string literal
    """

    def parse(self, arg: str) -> cp.Res[Union[Literal["auto"], float]]:
        arg = arg.strip().lower()
        if arg == "auto":
            return "auto"
        else:
            try:
                return float(arg)
            except ValueError as e:
                return cp.Err.make(str(e))


def mask_telluric_parser() -> cpp.Parser[Sequence[Tuple[float, float]]]:
    """
    Returns a parser for a sequence of tellurics

    The syntax is ``[[123.4, 234.5], ... [345.6, 456.7]]``.
    """
    number = ps.regex(r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?").map(float)
    pair = (ps.string("[") >> ps.seq(number << ps.string(","), number) << ps.string("]")).map(
        lambda s: (s[0], s[1])
    )
    pairs = ps.string("[") >> pair.sep_by(ps.string(",")) << ps.string("]")
    return cpp.Parser.from_parsy_parser(pairs)


str2bool = cpp.Parser.from_mapping(
    {"true": True, "false": False},
    aliases={
        "yes": True,
        "t": True,
        "y": True,
        "1": True,
        "no": False,
        "f": False,
        "n": False,
        "0": False,
    },
)


@dataclass(frozen=True)
class RassineMain(cp.Config):

    #: Stretch the x and y axes ratio ('auto_X' available), see Appendix B of https://arxiv.org/pdf/2006.13098.pdf
    par_stretching: Annotated[
        Union[float, Auto],
        cp.Param.store(
            ParStretchingParser(),
            default="auto_0.5",
            long_flag_name="--par_stretching",
            short_flag_name="-p",
        ),
    ]

    # is this the master spectrum? should we accept CSV format?

    #: Relative path of the spectrum pickle/csv file to process TODO: for what?
    spectrum_name: Annotated[
        RelPath,
        cp.Param.store(RelPath.parser, long_flag_name="--spectrum_name", short_flag_name="-s"),
    ]

    #: Directory where output files are written
    output_dir: Annotated[RelPath, cp.Param.store()]


rm = RassineMain.from_command_line_()
