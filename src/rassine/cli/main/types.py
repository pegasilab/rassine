"""Parsed types used in RASSINE configuration"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, Union

import configpile as cp
from typing_extensions import TypeAlias


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
