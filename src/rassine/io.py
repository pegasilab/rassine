import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union, overload

import numpy.typing as npt
import pandas as pd
from astropy.io import fits

default_pickle_protocol: int = 3

from typeguard import check_type


class NoCheck(Enum):
    TOKEN = 0


_T = TypeVar("_T")


@overload
def open_pickle(filename: Path, expected_type: Type[_T]) -> _T:
    pass


@overload
def open_pickle(filename: Path, expected_type: NoCheck = NoCheck.TOKEN) -> Any:
    pass


def open_pickle(filename: Path, expected_type: Any = NoCheck.TOKEN) -> Any:
    """
    Load information from a pickle

    Args:
        filename: Pickle to read
        expected_type: Optional type to validate the data read from

    Raises:
        TypeError: if there is a type mismatch

    Returns:
        Information read
    """
    logging.debug("Reading pickle %s", filename)
    res = pd.read_pickle(filename)
    if not isinstance(expected_type, NoCheck):
        check_type("pickle", res, expected_type)
    return res


def save_pickle(filename: Path, output: Any, protocol: int = default_pickle_protocol):
    """
    Save a pickle file with the proper protocol pickle version.

    Args:
        filename: Name of the output pickle file.
        output: Data to save
        protocol:
    """
    logging.debug("Writing pickle %s", filename)
    with open(filename, "wb") as f:
        pickle.dump(output, f, protocol=protocol)
