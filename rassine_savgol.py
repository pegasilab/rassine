import argparse
from pathlib import Path

import rassine as ras


def get_parser():
    """
    Returns the argument parser used in this script
    """
    res = argparse.ArgumentParser(
        description="""\
    SAVGOL step
    """
    )
    res.add_argument("inputfiles", type=Path, nargs="+", help="Input files to process")
    res.add_argument("--anchor-file", type=Path, required=True, help="Anchor file")
    res.add_argument(
        "--savgol-window", type=int, default=200, help="Length of the window for SAVGOL filtering"
    )
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ras.matching_diff_continuum(
        list(map(str, args.inputfiles)),
        sub_dico="matching_anchors",
        master=str(args.anchor_file),
        savgol_window=args.savgol_window,
        zero_point=False,
    )
