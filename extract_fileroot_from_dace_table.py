import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def get_parser():
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(description="""\
    DACE table filename extraction tool
    
    This tool takes the path to a CSV file as an argument and returns the contents of the
    fileroot column, which are paths, one per line.
    """)
    res.add_argument('inputfile', type=Path, help='CSV input filename')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    files = np.sort(pd.read_csv(args.inputfile)['fileroot'])
    for f in files:
        print(f)
