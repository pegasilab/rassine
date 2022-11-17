"""
Implementation of the matching_anchors_filter script
"""
from __future__ import annotations

import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, cast

import configpile as cp
import numpy as np
from filelock import FileLock
from typing_extensions import Annotated

from ..lib.data import LoggingLevel, NameRow, PathPattern, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.math import local_max, make_continuum
from ..lib.util import log_task_name_and_time
from ..rassine.data import RassinePickle
from .data import (
    AnchorOutput,
    AnchorParameters,
    AnchorPickle,
    MasterToolPickle,
    MatchingAnchorsRow,
)


def cast_as_anchor_pickle(arg: Any) -> AnchorPickle:
    # to make pylance happy
    return cast(AnchorPickle, arg)


def match_nearest(array1, array2):
    """
    Match the closest elements of two arrays vectors and return the matching matrix.

    Parameters
    ----------
    array1 : array_like
        First vector.
    array2 : array_like
        Second vector.

    Returns
    -------

    matching_matrix : array_like
        Matrix where each column contain :
        1) the indices in the first vector
        2) the indices in the second vector
        3) the values in the first vector
        4) the values in the second vector
        5) the distance between the closest elements

    """

    dmin = np.diff(np.sort(array1)).min()
    dmin2 = np.diff(np.sort(array2)).min()
    array1_r = array1 + 0.001 * dmin * np.random.randn(len(array1))
    array2_r = array2 + 0.001 * dmin2 * np.random.randn(len(array2))
    m = abs(array2_r - array1_r[:, np.newaxis])
    arg1 = np.argmin(m, axis=0)
    arg2 = np.argmin(m, axis=1)
    mask = np.arange(len(arg1)) == arg2[arg1]
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[liste_idx1]
    array2_k = array2[liste_idx2]
    return np.hstack(
        [
            liste_idx1[:, np.newaxis],
            liste_idx2[:, np.newaxis],
            array1_k[:, np.newaxis],
            array2_k[:, np.newaxis],
            (array1_k - array2_k)[:, np.newaxis],
        ]
    )


def intersect_all_continuum_single(
    filepath: Path,
    name: str,
    is_master: bool,
    tool: MasterToolPickle,
    tool_name: str,
    add_new: bool,
) -> Optional[MatchingAnchorsRow]:
    """
    Perform the intersection of the RASSINE files by using the anchor location saved
    in a master tool file.

    Args:
        filepath: Path of the file to process
        name: Spectrum/master spectrum name
        is_master: Whether this is a master spectrum file
        tool: Master tool data
        tool_name: Master tool name, used to avoid processing a file twice
        add_new: Add anchor points that were not detected.
    """

    fraction = tool["fraction"]
    tolerance = tool["tolerance"]
    threshold = tool["threshold"]
    copies_master = tool["nb_copies_master"]
    master_spectrum = tool["master_filename"]
    border2 = tool["border"]
    cluster_center = (border2[:, 1] - border2[:, 0]) / 2 + border2[:, 0]
    curve = tool["curve"]

    file: Union[RassinePickle, AnchorPickle] = open_pickle(filepath)
    if "matching_anchors" in file:
        file_as_ap: AnchorPickle = cast(AnchorPickle, file)
        if file_as_ap["matching_anchors"]["parameters"]["master_tool"] == tool_name:
            return
    spectrei = file["flux"]
    spectre = file["flux_used"]
    grid = file["wave"]

    index = file["output"]["anchor_index"]
    wave = file["output"]["anchor_wave"]
    flux = file["output"]["anchor_flux"]

    save = index.copy()

    diff = np.min([np.diff(save[1:]), np.diff(save[0:-1])], axis=0)
    diff = np.array([diff[0]] + list(diff) + [diff[-1]])
    diff = diff * float(fraction)
    diff = diff.astype("int")
    mask = np.zeros(len(grid))
    new_ = []
    for k in range(len(save)):
        new_.append(save[k] + np.arange(-diff[k], diff[k]))
    new = np.unique(np.hstack(new_))
    new = new[(new > 0) & (new < len(mask))]
    mask[new.astype("int")] = 1

    test = mask * curve

    mask_idx = test[save].astype("bool")
    mask_idx[0 : file["parameters"]["number_of_cut"]] = True
    mask_idx[-file["parameters"]["number_of_cut"] :] = True

    flux_denoised = flux

    save_before = len(index)

    index = index[mask_idx]
    wave = wave[mask_idx]
    flux = flux[mask_idx]
    flux_denoised = flux_denoised[mask_idx]

    save_after = len(index)

    index2 = (index >= border2[:, 0][:, np.newaxis]) & (index <= border2[:, 1][:, np.newaxis])
    cluster_empty = np.where(np.sum(index2, axis=1) == 0)[0]
    cluster_twin = np.where(np.sum(index2, axis=1) > 1)[0]

    if len(cluster_twin) != 0:
        index3 = np.unique(index * index2[cluster_twin, :])[1:]
        # centers_with_twin = cluster_center[cluster_twin]

        index_kept = index3[
            match_nearest(cluster_center, index3)[:, 1].astype("int")
        ]  # only twin closest to the cluster is kept
        index_to_suppress = np.setdiff1d(index3, index_kept)
        mask_idx = ~np.in1d(index, index_to_suppress)

        index = index[mask_idx]
        wave = wave[mask_idx]
        flux = flux[mask_idx]
        flux_denoised = flux_denoised[mask_idx]

    save_after_twin = len(index)

    if add_new:
        index_max, flux_max = local_max(spectre, file["parameters"]["vicinity_local_max"])

        new_max_index = []
        new_max_flux = []
        new_max_flux_denoised = []
        new_max_wave = []

        for k in cluster_empty:
            kept = (index_max >= border2[k, 0]) & (index_max <= border2[k, 1])
            if sum(kept) != 0:
                maxi = flux_max[kept].argmax()
                new_max_index.append((index_max[kept].astype("int"))[maxi])
                new_max_flux.append((flux_max[kept])[maxi])
                new_max_wave.append(grid[(index_max[kept].astype("int"))[maxi]])
                new_max_flux_denoised.append(
                    np.mean(
                        spectre[
                            (index_max[kept].astype("int"))[maxi]
                            - int(file["parameters"]["denoising_dist"]) : (
                                index_max[kept].astype("int")
                            )[maxi]
                            + int(file["parameters"]["denoising_dist"])
                            + 1
                        ]
                    )
                )

        new_max_index = np.array(new_max_index)
        new_max_flux = np.array(new_max_flux)
        new_max_flux_denoised = np.array(new_max_flux_denoised)
        new_max_wave = np.array(new_max_wave)

        index = np.hstack([index, new_max_index])
        wave = np.hstack([wave, new_max_wave])
        flux = np.hstack([flux, new_max_flux])
        flux_denoised = np.hstack([flux_denoised, new_max_flux_denoised])

    save_after_new = len(index)
    logging.info(
        "Result %s: %d/%d/%d/%d (before/after out-of-cluster/after twin/after adding)",
        name,
        save_before,
        save_after,
        save_after_twin,
        save_after_new,
    )

    continuum1, _, _, _ = make_continuum(
        wave,
        flux,
        flux_denoised,
        grid,
        spectrei,
        continuum_to_produce=("linear", "undenoised"),
    )

    index = index.astype("int")

    anchor_parameters: AnchorParameters = {
        "master_tool": tool_name,
        "master_filename": master_spectrum,
        "sub_dico_used": "output",
        "nb_copies_master": copies_master,
        "threshold": threshold,
        "tolerance": tolerance,
        "fraction": fraction,
    }
    anchor_output: AnchorOutput = {
        "continuum_linear": continuum1,
        "anchor_wave": wave,
        "anchor_flux": flux,
        "anchor_index": index,
        "parameters": anchor_parameters,
    }
    output: AnchorPickle = {
        "wave": file["wave"],
        "flux": file["flux"],
        "flux_err": file["flux_err"],
        "flux_used": file["flux_used"],
        "output": file["output"],
        "parameters": file["parameters"],
        "matching_anchors": anchor_output,
    }
    save_pickle(filepath, output)
    return MatchingAnchorsRow(
        name=name,
        is_master=is_master,
        n_anchors_before=save_before,
        n_anchors_ooc=save_after,
        n_anchors_twin=save_after_twin,
        n_anchors_adding=save_after_new,
    )


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Intersection tool, 2nd step

    Perform the intersection of the RASSINE files by using the anchor location saved in the master RASSINE spectrum.
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

    #: Path to master tool file produced by matching_anchors_scan
    master_tool: Annotated[Path, cp.Param.store(cp.parsers.path_parser)]

    #: Path of the master spectrum file to process
    #:
    #: The master spectrum is processed independently from other spectra
    #:
    #: Note that process_master is a path relative to "root"
    process_master: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-s", default_value=""
        ),
    ]

    #: Table of stacked spectra from which to read the individual spectrum names
    #:
    #: The parameters process_table, process_pattern and process_indices must be all provided together
    process_table: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-I", default_value=""
        ),
    ]

    #: Relative path (pattern) for the files to process
    #:
    #: The parameters process_table, process_pattern and process_indices must be all provided together
    process_pattern: Annotated[
        Optional[PathPattern],
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
            positional=cp.Positional.ZERO_OR_MORE,
            long_flag_name=None,
            short_flag_name=None,
        ),
    ]

    def validate_file(self) -> Optional[cp.Err]:
        """Validates the file parameters"""
        errors: List[Optional[cp.Err]] = []
        if self.process_indices:
            errors.append(
                cp.Err.check(
                    self.process_table is not None,
                    "If process_indices are provided, process_table must be provided",
                )
            )
            errors.append(
                cp.Err.check(
                    self.process_pattern is not None,
                    "If process_indices are provided, process_pattern must be provided",
                )
            )
        else:
            errors.append(
                cp.Err.check(
                    self.process_master is not None,
                    "At least one of process_indices or process_master must be provided",
                )
            )
        return cp.Err.collect(*errors)

    #: Add anchor points that were not detected.
    add_new: Annotated[bool, cp.Param.store(cp.parsers.bool_parser, default_value="true")]

    #: Diagnostic information table
    output_table: Annotated[
        Optional[Path],
        cp.Param.store(
            cp.parsers.path_parser.empty_means_none(), short_flag_name="-O", default_value=""
        ),
    ]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()
    diagnostic_rows: List[MatchingAnchorsRow] = []

    tool = open_pickle(t.root / t.master_tool, MasterToolPickle)
    tool_name = str(t.master_tool)

    if t.process_master is not None:
        logging.info("Processing master spectrum %s", t.process_master)
        res = intersect_all_continuum_single(
            filepath=t.root / t.process_master,
            name=str(t.process_master),
            is_master=True,
            tool=tool,
            tool_name=tool_name,
            add_new=t.add_new,
        )
        if res is not None:
            diagnostic_rows.append(res)

    if t.process_indices:
        logging.info("Processing %d stacked spectra provided", len(t.process_indices))
        assert t.process_table is not None
        assert t.process_pattern is not None
        tyble = NameRow.schema().read_csv(t.root / t.process_table, return_type="Tyble")
        for i in t.process_indices:
            row = tyble[i]
            res = intersect_all_continuum_single(
                filepath=t.process_pattern.to_path(t.root, row.name),
                name=row.name,
                is_master=False,
                tool=tool,
                tool_name=tool_name,
                add_new=t.add_new,
            )
            if res is not None:
                diagnostic_rows.append(res)

    if t.output_table is not None:
        output_table_path = t.root / t.output_table
        output_table_lockfile = output_table_path.with_suffix(output_table_path.suffix + ".lock")
        logging.debug("Appending to output table %s", t.output_table)
        with FileLock(output_table_lockfile):
            df = MatchingAnchorsRow.schema().from_rows(diagnostic_rows, return_type="DataFrame")
            df.to_csv(
                output_table_path, header=not output_table_path.exists(), mode="a", index=False
            )


def cli():
    run(Task.from_command_line_())
