#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Optional, Union

import deepdish as dd
import numpy as np
import numpy.typing as npt

import rassine.io
from rassine.functions import preprocess_prematch_stellar_frame


def float_or_rv_file(s: str) -> Union[npt.ArrayLike, float]:
    try:
        res = float(s)
        return res
    except ValueError:
        return rassine.io.read_rv(Path(s))


def get_parser():
    res = argparse.ArgumentParser(
        allow_abbrev=False,
        description="""\
    Identification tool for wavelength vector parameters
    
    Define the wavelength vector on which all the spectra will be reinterpolated (common wavelength vector) and detect CDD 
    gap.
    
    The wavelength vector is defined from the maximum of all the spectra minimum wavelength and minimum of all the spectra 
    maximum wavelength. 
    
    A RV shift can be applied on all the spectra during the reinterpolation stage.
    
    It processes all the *.p files present in the input directory.
    """,
    )
    res.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing the files to process"
    )
    res.add_argument(
        "--dlambda", "-d", type=float, default=None, help="Instrument format of the s1d spectra"
    )
    res.add_argument(
        "--no-dlambda",
        dest="dlambda",
        action="store_const",
        const=None,
        help="Set dlambda to None",
    )
    res.add_argument(
        "--rv",
        "-k",
        type=float_or_rv_file,
        default="0.0",
        help="RV shift in kms to apply on the spectra (e.g binary, large trend), either a float, the path to a pickle file, the path to a CSV file",
    )
    res.add_argument(
        "--output-file", type=Path, required=True, help="Output HDF5 file containing the results"
    )
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    dlambda: Optional[float] = args.dlambda
    rv: Union[npt.ArrayLike, float] = args.rv
    output_file: Path = args.output_file
    inputfiles = np.sort(list(input_dir.glob("*.p")))  # type: ignore

    assert len(inputfiles) > 0, "At least one input file must be available"

    (
        wave_min_k,
        wave_max_k,
        dlambda,
        hole_left_k,
        hole_right_k,
        static_grid,
        wave_min,
        berv,
        lamp,
        plx_mas,
        acc_sec,
        rv,
        rv_mean,
    ) = preprocess_prematch_stellar_frame(list(map(str, inputfiles)), dlambda=dlambda, rv=rv)
    output = {
        "wave_min_k": wave_min_k,
        "wave_max_k": wave_max_k,
        "dlambda": dlambda,
        "hole_left_k": hole_left_k,
        "hole_right_k": hole_right_k,
        "static_grid": static_grid,
        "wave_min": wave_min,
        "berv": berv,
        "lamp": lamp,
        "plx_mas": plx_mas,
        "acc_sec": acc_sec,
        "rv": rv,
        "rv_mean": rv_mean,
    }
    dd.io.save(output_file, output)
