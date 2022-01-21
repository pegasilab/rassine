#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import deepdish as dd
from typing import List
from pathlib import Path
import numpy as np
from rassine import preprocess_match_stellar_frame
import argparse

def get_parser():
    res = argparse.ArgumentParser(allow_abbrev=False, description="""\
    Matching tool
    
    Rewrites spectra in place
    """)
    res.add_argument('indices', type=int, nargs='+',
                     help='0-based indices of files to process')
    res.add_argument('--input-dir', type=Path, required=True,
                     help='Directory containing the files to process')
    res.add_argument('--parameter-file', type=Path, required=True,
                     help='Parameter file (output of rassine_preprocess_match_stellar_frame.py)')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    indices: List[int] = args.indices
    input_dir: Path = args.input_dir
    allfiles = np.sort(list(map(str, input_dir.glob('*.p'))))
    parameter_file: Path = args.parameter_file
    p = dd.io.load(parameter_file)
    args = (p['wave_min_k'], p['wave_max_k'], p['dlambda'], p['hole_left_k'], p['hole_right_k'],
            p['static_grid'], p['wave_min'][indices], p['berv'][indices], p['lamp'][indices],
            p['plx_mas'][indices], p['acc_sec'][indices], p['rv'][indices], p['rv_mean'])
    # what about dlambda
    preprocess_match_stellar_frame(allfiles[indices], args)
