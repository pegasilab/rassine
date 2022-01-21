import argparse
import rassine as ras
from pathlib import Path


def get_parser():
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(description="""\
    Stacking tool
                                        
    Stacks all the spectra in a directory according to a defined binning length.
    
    The spectra are understood as all the .p files in the given directory; the binned spectra
    are written in the ../STACKED directory relative to the given directory.
    """)

    res.add_argument('--bin_length_stack', type=int, default=1,
                     help='Length of the binning for the stacking in days')
    res.add_argument('--input_directory', type=Path,
                     help='Path to the directory containing the input files')
    res.add_argument('--dbin', type=float, default=0.0,
                     help='dbin to shift the binning (0.5 for solar data)')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    files = args.input_directory.glob('*.p')
    master_name = ras.preprocess_stack(list(map(str, files)),
                                       bin_length=args.bin_length_stack,
                                       dbin=args.dbin,
                                       make_master=True)
