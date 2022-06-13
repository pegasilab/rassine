from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def grouping(
    array: NDArray[np.float64], threshold: float, num: int
) -> Tuple[Sequence[NDArray[np.float64]], NDArray[np.int64]]:
    difference = abs(np.diff(array))
    cluster = difference < threshold
    indices = np.arange(len(cluster))[cluster]

    j = 0
    border_left = [indices[0]]
    border_right = []
    while j < len(indices) - 1:
        if indices[j] == indices[j + 1] - 1:
            j += 1
        else:
            border_right.append(indices[j])
            border_left.append(indices[j + 1])
            j += 1
    border_right.append(indices[-1])
    border = np.array([border_left, border_right]).T
    border = np.hstack([border, (1 + border[:, 1] - border[:, 0])[:, np.newaxis]])

    kept: List[NDArray[np.float64]] = []
    for j in range(len(border)):
        if border[j, -1] >= num:
            kept.append(array[border[j, 0] : border[j, 1] + 2])
    return kept, border


def find_nearest1(
    array: NDArray[np.float64], value: float, dist_abs: bool = True
) -> Tuple[int, float, float]:
    """
    Find the closest element of a vector

    Note:
        Compared to the original find_nearest, it takes a scalar "value"

    Args:
        array: Vector
        value: Value to find
        dist_abs: Whether the return the distance in absolute value
    Returns:
        A tuple (index, value, distance) of the closest element
    """
    idx = int(np.argmin(np.abs(array - value)))
    if dist_abs:
        distance = abs(array[idx] - value)
    else:
        distance = array[idx] - value
    return idx, array[idx], distance


def rolling_iq(
    array: NDArray[np.float64], window: int = 1, min_periods: int = 1
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform a rolling IQ statistic in a fixed window.

    Args:
        array: The vector to investigate.
        window: The window used for the rolling statistic.
        min_periods: Computation of the statistics up to the min_periods border value

    Returns
    -------
    rolling_Q1S:
        The rolling 25th percentile.
    rolling_Q3:
        The rolling 75th percentile.
    rolling_IQ:
        The rolling IQ (Q3-Q1).
    """
    roll_Q1 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.25)
    )
    roll_Q3 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.75)
    )
    roll_IQ = roll_Q3 - roll_Q1  # type: ignore
    return roll_Q1, roll_Q3, roll_IQ
