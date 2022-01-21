#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:41:21 2019

@author: cretignier
"""

from __future__ import print_function
import argparse
from rassine import preprocess_fits
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(allow_abbrev=False, description="""\
    RASSINE preprocessing tool
    
    Preprocess the files spectra to produce readeable RASSINE files.
        
    The preprocessing depends on the s1d format of the instrument : HARPS, HARPN, CORALIE or ESPRESSO 
    (if new DRS is used, use the ESPRESSO kw not matter the instrument)
    """)

    res.add_argument('inputfiles', type=Path, nargs='+',
                        help='Input files to process')
    res.add_argument('--instrument', '-i', type=str, default='HARPS',
                        help='Instrument format of the s1d spectra')
    res.add_argument('--plx-mas', type=float, default=0.0,
                        help='parallaxe in mas (no more necessary ?)')
    res.add_argument('--output-dir', '-o', type=str, default='',
                        help='Name of the output directory. If None, the output directory is created at the same location than the spectra.')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    inputfiles = args.inputfiles

    assert len(inputfiles) > 0, 'At least one input file must be specified'
    output_dir = args.output_dir
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = inputfiles[0].parent

    preprocess_fits(list(map(str, inputfiles)), instrument=args.instrument, plx_mas=args.plx_mas, final_sound=False, output_dir=str(output_dir))
