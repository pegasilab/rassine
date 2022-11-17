import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import configpile as cp
import rich.pretty
from recursive_diff import recursive_diff
from typing_extensions import Annotated


@dataclass(frozen=True)
class Task(cp.Config):

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

    #: Subdico to inspect, several keys may be separated by dots
    key: Annotated[str, cp.Param.store(cp.parsers.stripped_str_parser, default_value="")]


def run(t: Task):
    with open(t.root / t.input1, "rb") as file:
        data1 = pickle.load(file)
    with open(t.root / t.input2, "rb") as file:
        data2 = pickle.load(file)
    subkeys = [s.strip() for s in t.key.split(".") if s.strip() != ""]
    for key in subkeys:
        data1 = data1[key]
        data2 = data2[key]
    rich.pretty.pprint(list(recursive_diff(data1, data2)))


def cli():
    run(Task.from_command_line_())
