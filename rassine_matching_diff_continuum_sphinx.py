import argparse
import rassine as ras
import textwrap
import os
import glob
from pathlib import Path


def get_parser():
    res = argparse.ArgumentParser(description="""\
    Continuum matching tool
    
    Matches the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference. 
    The savogol window parameter can be selected by the GUI interface. 
    
    The input files are the ones matching the RASSINE*.p pattern in the given directory.
    """)

    res.add_argument('--input_directory', type=Path,
                        help='Path to the directory containing the input files')
    res.add_argument('--master', type=str, default=None,
                        help='Name of the RASSINE master spectrum file'
                        )
    res.add_argument('--no-master', dest='master', action='store_const', const=None)
    res.add_argument('--sub_dico', type=str, choices=['output', 'matching_anchors'], default='matching_anchors',
                        help="Name of the continuum to use. Either 'output' (RASSINE individual) or 'matching_anchors' (RASSINE time-series)")
    res.add_argument('--savgol_window', type=int, default=200,
                        help='Length of the window for the savgol filtering')
    res.add_argument('--zero_point', type=bool, default=False,
                        help='No more used ?')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # preprocess_stack wants strings
    files = [str(path) for path in args.input_directory.glob('RASSINE*.p')]
    master, savgol_window = ras.matching_diff_continuum_sphinx(files, sub_dico=args.sub_dico, master=args.master, savgol_window=args.savgol_window, zero_point=args.zero_point)
    print(' ')
    print(savgol_window)
