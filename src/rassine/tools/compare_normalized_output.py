import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import configpile as cp
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Annotated, Literal

from ..lib.io import open_pickle
from ..matching.data import MatchingPickle


@dataclass(frozen=True)
class Task(cp.Config):

    #
    # Common information
    #

    prog_ = Path(__file__).stem
    ini_strict_sections_ = [Path(__file__).stem]
    ini_relaxed_sections_ = [Path(__file__).stem.split("_")[0]]
    env_prefix_ = "RASSINE"

    #
    # Task specific information
    #

    #: First pickle to compare
    input1: Annotated[
        Path,
        cp.Param.store(cp.parsers.path_parser, positional=cp.Positional.ONCE, long_flag_name=None),
    ]

    #: Second pickle to compare
    input2: Annotated[
        Path,
        cp.Param.store(cp.parsers.path_parser, positional=cp.Positional.ONCE, long_flag_name=None),
    ]

    #: Kind of output to compare
    kind: Annotated[
        Literal["output", "matching_anchors", "matching_diff"],
        cp.Param.store(cp.Parser.from_choices(["output", "matching_anchors", "matching_diff"])),
    ]

    #: Threshold for acceptance in normalized units
    threshold: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="0.005")]


def get_continuum_linear(
    data: MatchingPickle, kind: Literal["output", "matching_anchors", "matching_diff"]
) -> NDArray[np.float64]:
    """Returns the flux data corresponding to the given key

    Args:
        data: RASSINE output data
        kind: Kind of spectrum to get
    """
    if kind == "output":
        return data["output"]["continuum_linear"]
    elif kind == "matching_anchors":
        return data["matching_anchors"]["continuum_linear"]
    elif kind == "matching_diff":
        return data["matching_diff"]["continuum_linear"]


def compute_distance(
    data1: MatchingPickle,
    data2: MatchingPickle,
    kind: Literal["output", "matching_anchors", "matching_diff"],
) -> np.float64:
    """Returns the distance between two outputs

    Args:
        data1: RASSINE output data (1st)
        data2: RASSINE output data (2nd)
        kind: Kind of output used for comparison
    """
    return np.std(
        data1["flux"] / get_continuum_linear(data1, kind)
        - data2["flux"] / get_continuum_linear(data2, kind)
    )


def run(t: Task):
    data1 = cast(MatchingPickle, open_pickle(t.input1))
    data2 = cast(MatchingPickle, open_pickle(t.input2))
    distance = compute_distance(data1, data2, t.kind)
    if distance <= t.threshold:
        print(f"Distance below threshold: {distance} <= {t.threshold}")
        sys.exit(0)
    else:
        print(f"Distance above threshold: {distance} > {t.threshold}")
        sys.exit(1)


def cli():
    run(Task.from_command_line_())
