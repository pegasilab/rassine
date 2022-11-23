import argparse
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence

import configpile as cp
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Annotated

from ..lib.analysis import grouping
from ..lib.data import LoggingLevel, PathPattern, PickleProtocol
from ..lib.io import open_pickle, save_pickle
from ..lib.util import log_task_name_and_time
from ..rassine.data import RassinePickle
from ..stacking.data import StackedBasicRow
from .data import MasterToolPickle


def intersect_all_continuum_sphinx(
    names: Sequence[Path],
    output_file: Path,
    master_spectrum: Optional[Path] = None,
    copies_master: int = 0,
    fraction: float = 0.2,
    threshold: float = 0.66,
    tolerance: float = 0.5,
):
    """
    Search for the intersection of all the anchors points in a list of filename and update the
    selection of anchor points in the same files.

    For each anchor point the fraction of the closest distance to a neighborhood is used.
    Anchor point under the threshold are removed (1 = most deleted, 0 = most kept).
    Possible to use multiprocessing with nthreads cpu.
    If you want to fix the anchors points, enter a master spectrum path and the number of copies
    you want of it.

    Args:
        names: List of RASSINE files.
        output_file: Master tool file to write
        master_spectrum: Name of the RASSINE master spectrum file.
        copies_master: Number of copies of the master, to weigh more the master data
                       during the analysis.

                       If 0 value is specified, copies_master is set to 2*N
                       with N the number of RASSINE files.
        fraction: Parameter of the model between 0 and 1
        treshold: Parameter of the model between 0 and 1
        tolerance: Parameter of the model between 0 and 1
    """

    logging.info("Loading of the files, wait ...")

    wave = open_pickle(names[0], RassinePickle)["wave"]
    save: List[NDArray[np.int64]] = [
        open_pickle(filename, RassinePickle)["output"]["anchor_index"] for filename in names
    ]

    if master_spectrum is not None:
        if copies_master == 0:
            logging.warning(
                "You have to specify the number of copy of the master file you want as argument."
            )
            copies_master = 2 * len(names)

            logging.warning("Default value of master copies fixed at %d", copies_master)
        file: RassinePickle = open_pickle(master_spectrum, RassinePickle)
        for j in range(copies_master):
            save.append(file["output"]["anchor_index"])
    logging.info("Got %d files", len(save))
    sum_mask = []
    all_idx = np.hstack(save)

    logging.info("Computation of the intersection of the anchors points, wait ...")
    for j in range(len(names)):
        diff = np.min([np.diff(save[j][1:]), np.diff(save[j][0:-1])], axis=0)
        diff = np.array([diff[0]] + list(diff) + [diff[-1]])
        diff = diff * fraction
        diff = diff.astype("int")
        mask = np.zeros(len(wave))
        new = []
        for k in range(len(save[j])):
            new.append(save[j][k] + np.arange(-diff[k], diff[k]))
        new = np.unique(np.hstack(new))
        new = new[(new > 0) & (new < len(mask))]
        mask[new.astype("int")] = 1
        sum_mask.append(mask)
    sum_mask = np.array(sum_mask)
    sum_mask_vert = np.sum(sum_mask, axis=0)

    strat = np.linspace(int(sum_mask_vert.min()), int(sum_mask_vert.max()), 10).astype("int")
    strat = strat[strat != strat[1]]  # suppress the first level

    for j in range(len(strat) - 1)[::-1]:
        sum_mask_vert[(sum_mask_vert >= strat[j]) & (sum_mask_vert < strat[j + 1])] = strat[j]
    for j in range(len(strat[0:-1]))[::-1]:
        sum_mask_vert[sum_mask_vert == strat[0:-1][j]] = strat[j + 1]

    # sum_mask_vert -= np.diff(strat)[0]
    sum_mask_vert[sum_mask_vert == np.unique(sum_mask_vert)[0]] = 0

    mask_vert2 = sum_mask_vert.copy()
    mask_vert2[0] = 0
    mask_vert2[-1] = 0

    for j in range(len(mask_vert2) - 2):
        j += 1
        if (mask_vert2[j] > mask_vert2[j - 1]) | (mask_vert2[j] > mask_vert2[j + 1]):
            # supression of delta peak (not possible because all the peaks cannot be situated
            # #at the exact same wavelength), allow the use of grouping function here after
            mask_vert2[j] = np.max([mask_vert2[j - 1], mask_vert2[j + 1]])

    val, border = grouping(mask_vert2, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    null = np.where(border[:, -1] == 0)[0]

    area = []
    small_length = []
    small_center = []
    for j in range(len(null) - 1):
        area.append(
            np.sum(border[null[j] + 1 : null[j + 1], 2] * border[null[j] + 1 : null[j + 1], 3])
        )
        peak = np.where(
            border[null[j] + 1 : null[j + 1], 3] == (border[null[j] + 1 : null[j + 1], 3].max())
        )[0]
        peak = peak[border[null[j] + 1 : null[j + 1], 2][peak].argmax()]
        small_length.append(border[null[j] + 1 : null[j + 1], 2][peak])
        small_center.append(border[null[j] + 1 : null[j + 1], 0][peak] + small_length[-1] / 2)

    left = border[null, 1][0:-1]
    right = border[null, 0][1:]

    center2_: List[int] = []
    for i, j in zip(left, right):
        slc = all_idx[(all_idx >= i) & (all_idx <= j)]
        if len(slc):
            center2_.append(int(np.median(slc)))
        else:
            center2_.append(0)
    center2 = np.array(center2_, dtype="int64")  # kept
    center2[center2 < 0] = 0

    large_length = right - left
    large_center = left + large_length / 2

    center = np.mean(np.array([[small_center], [large_center]]), axis=0)[0]
    windows = np.mean(np.array([[small_length], [large_length]]), axis=0)[0]
    height = area / windows

    center = np.round(center, 0).astype("int")
    windows = np.round(windows / 2, 0).astype("int")
    height = np.round(height, 0).astype("int")

    center = center2

    # take any scan
    gri = wave

    stack_vert3 = np.zeros(len(sum_mask_vert))

    stack_vert = np.zeros(len(sum_mask_vert))
    for j in range(len(save)):  # pylint: disable=C0200
        stack_vert[save[j]] += 1

    for j in range(len(center)):  # pylint: disable=C0200
        if center[j] != np.nan:
            stack_vert3[center[j] - windows[j] : center[j] + windows[j] + 1] = height[j]

    stack_vert3[stack_vert3 < (len(names) * threshold)] = 0
    stack_vert3[stack_vert3 > (len(names))] = len(names)
    val, border = grouping(stack_vert3, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    for j in range(len(border)):
        if np.sum(stack_vert[int(border[j, 0]) : int(border[j, 1]) + 1]) < (
            border[j, 3] * tolerance
        ):
            stack_vert3[int(border[j, 0]) : int(border[j, 1]) + 1] = 0

    liste_ = []
    for j in range(len(stack_vert3) - 2):
        j += 1
        if (stack_vert3[j] != stack_vert3[j - 1]) | (stack_vert3[j] != stack_vert3[j + 1]):
            liste_.append(j + 1)
    liste = np.array(liste_)
    if len(liste) != 0:
        stack_vert3[liste] = 0

    val, border = grouping(stack_vert3, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    val2, border2 = grouping(stack_vert3, 1, 1)
    border2 = np.hstack([border2, np.array([i[0] for i in val2])[:, np.newaxis]])
    border2 = border2[border2[:, -1] > 0]
    curve = stack_vert3 > 0

    output_cluster: MasterToolPickle = {
        "curve": curve,
        "border": border2.astype(np.float64),
        "wave": gri,
        "threshold": threshold,
        "tolerance": tolerance,
        "fraction": fraction,
        "nb_copies_master": copies_master,
        "master_filename": None if master_spectrum is None else master_spectrum.name,
    }

    save_pickle(output_file, output_cluster)


@dataclass(frozen=True)
class Task(cp.Config):
    """
    Intersection tool

    Searches for the intersection of all the anchors points in a list of filenames and updates
    the selection of anchor points in the same files.

    For each anchor point the fraction of the closest distance to a neighbourhood is used.
    Anchor point under the threshold are removed (1 = most deleted, 0 = most kept).
    Possible to use multiprocessing with nthreads cpu.
    If you want to fix the anchors points, enter a master spectrum path and the number of
    copies you want of it.

    It reads the files matching RASSINE*.p in the given input directory.
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

    #: Table of spectra from which to read the individual file names (produced by stacking step)
    input_table: Annotated[
        Path,
        cp.Param.store(cp.parsers.path_parser, short_flag_name="-I"),
    ]

    #: Path pattern for input files
    input_pattern: Annotated[
        PathPattern,
        cp.Param.store(PathPattern.parser(), short_flag_name="-i", default_value="RASSINE_{}.p"),
    ]

    #: Master tool output file
    output_file: Annotated[Path, cp.Param.store(cp.parsers.path_parser, short_flag_name="-o")]

    #: Name of the RASSINE master spectrum file
    master_spectrum: Annotated[
        Optional[Path], cp.Param.store(cp.parsers.path_parser.empty_means_none(), default_value="")
    ]

    #: Do not use a RASSINE master spectrum file
    no_master_spectrum: ClassVar[cp.Expander] = cp.Expander.make(
        new_flag="--master-spectrum", new_value=""
    )

    #: Number of copy of the master.
    #:
    #: If value 0 is specified, copies_master is set to 2*N with N the number of RASSINE files.
    copies_master: Annotated[int, cp.Param.store(cp.parsers.int_parser, default_value="0")]

    #: Unused: number of threads for multiprocessing
    nthreads: Annotated[int, cp.Param.store(cp.parsers.int_parser, default_value="1")]

    #: Parameter of the model between 0 and 1
    fraction: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="0.2")]

    #: Parameter of the model between 0 and 1
    threshold: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="0.65")]

    #: Parameter of the model between 0 and 1
    tolerance: Annotated[float, cp.Param.store(cp.parsers.float_parser, default_value="0.5")]


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for Sphinx doc purposes"""
    return Task.get_argument_parser_()


@log_task_name_and_time(name=Path(__file__).stem)
def run(t: Task) -> None:
    t.logging_level.set()
    t.pickle_protocol.set()

    input_tyble = StackedBasicRow.schema().read_csv(t.root / t.input_table, return_type="Tyble")
    files = [t.input_pattern.to_path(t.root, r.name) for r in input_tyble]

    intersect_all_continuum_sphinx(
        files,
        master_spectrum=(None if t.master_spectrum is None else t.root / t.master_spectrum),
        copies_master=t.copies_master,
        fraction=t.fraction,
        threshold=t.threshold,
        tolerance=t.tolerance,
        output_file=t.root / t.output_file,
    )


def cli():
    run(Task.from_command_line_())
