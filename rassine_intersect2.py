import argparse
import rassine as ras
from pathlib import Path


def get_parser():
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(description="""\
    Intersection tool, 2nd step
                                        
    Perform the intersection of the RASSINE files by using the anchor location saved in the master RASSINE spectrum. 
    """)

    res.add_argument('names', type=Path, nargs='+',
                     help='List of RASSINE files. ')
    res.add_argument('--add_new', type=bool, default=True,
                     help='Add anchor points that were not detected.')
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ras.intersect_all_continuum(list(map(str, args.names)), add_new=args.add_new)
