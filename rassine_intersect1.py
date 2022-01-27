import argparse
import rassine as ras
import textwrap
import os
import glob
from pathlib import Path

def get_parser():
    """
    Returns the argument parser used in this script
    """
    parser = argparse.ArgumentParser(description="""\
    Intersection tool
    
    Searches for the intersection of all the anchors points in a list of filenames and updates the selection of anchor points in the same files.
    
    For each anchor point the fraction of the closest distance to a neighbourhood is used.
    Anchor point under the threshold are removed (1 = most deleted, 0 = most kept).
    Possible to use multiprocessing with nthreads cpu.
    If you want to fix the anchors points, enter a master spectrum path and the number of copies you want of it.
    
    It reads the files matching RASSINE*.p in the given input directory.
    """)

    parser.add_argument('--input-directory', type=Path,
                        help='Path to the directory containing the input files')
    parser.add_argument('--feedback', type=bool,
                        help='Activate the GUI Matplotlib interface')
    parser.add_argument('--master-spectrum', type=str, default=None,
                        help='Name of the RASSINE master spectrum file'
                        )
    parser.add_argument('--no-master-spectrum', dest='master_spectrum', action='store_const', const=None,
                        help='Do not use a RASSINE master spectrum file')
    parser.add_argument('--copies-master', type=int, default=0,
                        help='Number of copy of the master. If 0 value is specified, copies_master is set to 2*N with N the number of RASSINE files.')
    parser.add_argument('--kind', type=str, default='anchor_index',
                        help='Entry kw of the vector to select in the RASSINE file')
    parser.add_argument('--nthreads', type=int, default=1,
                        help='Number of threads for multiprocessing')
    parser.add_argument('--fraction', type=float, default=0.2,
                        help='Parameter of the model between 0 and 1')
    parser.add_argument('--threshold', type=float, default=0.66,
                        help='Parameter of the model between 0 and 1')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Parameter of the model between 0 and 1')
    parser.add_argument('--add-new', type=bool, default=True,
                        help='Add anchor points that were not detected. HAS NO EFFECT')
    return parser


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    # preprocess_stack wants strings
    files = [str(path) for path in args.input_directory.glob('RASSINE*.p')]
    ras.intersect_all_continuum_sphinx(files, feedback=args.feedback, master_spectrum=args.master_spectrum,
                                       copies_master=args.copies_master, kind=args.kind, nthreads=args.nthreads,
                                       fraction=args.fraction, threshold=args.threshold, tolerance=args.tolerance, add_new = args.add_new)
