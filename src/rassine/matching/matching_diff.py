"""Implements the matching_diff script"""
import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypedDict

import configpile as cp
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Annotated

from ..lib.analysis import rolling_iq
from ..lib.data import LoggingLevel, NameRow, PathPattern, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import smooth
from ..lib.util import log_task_name_and_time
from ..rassine.data import RassineBasicOutput, RassineParameters
from .data import (
    AnchorOutput,
    AnchorPickle,
    MatchingDiffOutput,
    MatchingDiffParameters,
    MatchingPickle,
)


def matching_diff_continuum(
    input_data: AnchorPickle,
    master_name: str,
    master_data: AnchorPickle,
    savgol_window: int = 200,
    zero_point: bool = False,
) -> MatchingPickle:
    """
    Match the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference.

    Creates the matching_diff dictionary

    Args:
        path: Path of the pickle file to process
        master_: Name of the RASSINE master spectrum file.
        savgol window: Length of the window for the savgol filtering.
        zero_point: No more used ?
    """
    sub_dico = "matching_anchors"
    dx = np.diff(master_data["wave"])[0]
    length_clip = int(100 / dx)  # smoothing on 100 \ang for the tellurics clean

    spectre = input_data["flux_used"]

    par: MatchingDiffParameters = {
        "reference_continuum": master_name,
        "savgol_window": savgol_window,
        "recenter": zero_point,
        "sub_dico_used": sub_dico,
    }

    cont = input_data[sub_dico]["continuum_linear"]
    cont2 = master_data["flux_used"] / master_data[sub_dico]["continuum_linear"]

    cont1 = spectre / cont
    diff = cont1 - cont2
    med_value = np.nanmedian(diff)
    for _ in range(3):  # needed to avoid artefact induced by tellurics
        q1, q3, iq = rolling_iq(diff, window=length_clip)
        diff[(diff > q3 + 1.5 * iq) | (diff < q1 - 1.5 * iq)] = med_value
        diff[diff < q1 - 1.5 * iq] = med_value

    correction = smooth(diff, savgol_window, shape="savgol")
    correction = smooth(correction, savgol_window, shape="savgol")
    if zero_point:
        correction = correction - np.nanmedian(correction)
    cont_corr: NDArray[np.float64] = cont.copy() / (1 - correction.copy())
    this_output: MatchingDiffOutput = {"parameters": par, "continuum_linear": cont_corr}
    return {
        "wave": input_data["wave"],
        "flux": input_data["flux"],
        "flux_err": input_data["flux_err"],
        "flux_used": input_data["flux_used"],
        "output": input_data["output"],
        "parameters": input_data["parameters"],
        "matching_anchors": input_data["matching_anchors"],
        "matching_diff": this_output,
    }


@dataclass(frozen=True)
class Task(cp.Config):
    """
    SAVGOL step

    Match the continuum of individual spectra to a reference spectrum
    with a savgol filtering on the spectra difference.

    Creates the matching_diff dictionary
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

    #: Table of stacked spectra from which to read the individual spectrum names
    #:
    #: The parameters process_table, process_pattern and process_indices must be all provided together
    process_table: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-I", default_value=""
        ),
    ]

    #: Relative path (pattern) for the files to process
    #:
    #: The parameters process_table, process_pattern and process_indices must be all provided together
    process_pattern: Annotated[
        PathPattern,
        cp.Param.store(
            PathPattern.parser().empty_means_none(), short_flag_name="-i", default_value=""
        ),
    ]

    #: Indices of spectrum to process
    #:
    #: The parameters process_table, process_pattern and process_indices must be all provided together
    process_indices: Annotated[
        Sequence[int],
        cp.Param.append1(
            cp.parsers.int_parser,
            positional=cp.Positional.ONE_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    #: Length of the window for SAVGOL filtering
    savgol_window: Annotated[int, cp.Param.store(cp.parsers.int_parser, default_value="200")]

    #: Processed master spectrum / Anchor file
    anchor_file: Annotated[Path, cp.Param.store(cp.parsers.path_parser)]

    #: Recenter to the mean value
    zero_point: Annotated[bool, cp.Param.store(cp.parsers.bool_parser, default_value="false")]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()

    master_path = t.root / t.anchor_file
    master_name = master_path.name
    master_data = open_pickle(master_path, AnchorPickle)

    logging.info("Processing %d stacked spectra provided", len(t.process_indices))
    tyble = NameRow.schema().read_csv(t.root / t.process_table, return_type="Tyble")
    for i in t.process_indices:
        file = t.process_pattern.to_path(t.root, tyble[i].name)
        input_data = open_pickle(file, AnchorPickle)
        output_data = matching_diff_continuum(
            input_data=input_data,
            master_name=master_name,
            master_data=master_data,
            savgol_window=t.savgol_window,
            zero_point=t.zero_point,
        )
        save_pickle(file, output_data)


def cli():
    run(Task.from_command_line_())
