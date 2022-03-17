from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import Float


def grouping(array: npt.NDArray[np.float64], tresh: Float, num: int):
    difference = abs(np.diff(array))
    cluster = difference < tresh
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

    kept = []
    for j in range(len(border)):
        if border[j, -1] >= num:
            kept.append(array[border[j, 0] : border[j, 1] + 2])
    return np.array(kept, dtype="object"), border


def clustering(array: npt.NDArray[np.float64], tresh: Float, num: Float):

    """
    Detect and form 1D-cluster on an array. A new cluster is formed once the next vector value is farther than a treshold value.

    Parameters
    ----------
    array : array_like
        The vector used to create the clustering (1D)
    tresh : float
        Threshold value distance used to define a new cluster.
    num : float
        The minimum number of elements to consider a cluster

    Returns
    -------
    cluster_matrix : array_like
        The matrix containing the left indice, right indice and the length of the 1D cluster

    """

    difference = np.diff(array)
    cluster = difference < tresh
    indice = np.arange(len(cluster))[cluster]
    if sum(cluster):
        j = 0
        border_left = [indice[0]]
        border_right = []
        while j < len(indice) - 1:
            if indice[j] == indice[j + 1] - 1:
                j += 1
            else:
                border_right.append(indice[j])
                border_left.append(indice[j + 1])
                j += 1
        border_right.append(indice[-1])
        border = np.array([border_left, border_right]).T
        border = np.hstack([border, (1 + border[:, 1] - border[:, 0])[:, np.newaxis]])

        kept: List[Any] = []
        for j in range(len(border)):
            if border[j, -1] >= num:
                kept.append(array[border[j, 0] : border[j, 1] + 2])
        return np.array(kept, dtype="object")
    else:
        return [np.array([j]) for j in array]


def find_nearest(array, value, dist_abs=True) -> Tuple[int, int, float]:
    """
    Find the closest element of a vector.

    Parameters
    ----------
    array : array_like
        Wavelength vector.
    value : array_like
        Flux vector.
    dist_abs : bool
        Give the distance in absolute value.

    Returns
    -------
    indice : int
        Indice of the closest element.
    value : int
        Value of the closest element.
    distance : float
        Distance to the closest element.

    """

    if type(value) != np.ndarray:
        value = np.array([value])
    idx = np.argmin((np.abs(array - value[:, np.newaxis])), axis=1)
    distance = abs(array[idx] - value)
    if dist_abs == False:
        distance = array[idx] - value
    return idx, array[idx], distance


def match_nearest(array1, array2):
    """
    Match the closest elements of two arrays vectors and return the matching matrix.

    Parameters
    ----------
    array1 : array_like
        First vector.
    array2 : array_like
        Second vector.

    Returns
    -------

    matching_matrix : array_like
        Matrix where each column contain :
        1) the indices in the first vector
        2) the indices in the second vector
        3) the values in the first vector
        4) the values in the second vector
        5) the distance between the closest elements

    """

    dmin = np.diff(np.sort(array1)).min()
    dmin2 = np.diff(np.sort(array2)).min()
    array1_r = array1 + 0.001 * dmin * np.random.randn(len(array1))
    array2_r = array2 + 0.001 * dmin2 * np.random.randn(len(array2))
    m = abs(array2_r - array1_r[:, np.newaxis])
    arg1 = np.argmin(m, axis=0)
    arg2 = np.argmin(m, axis=1)
    mask = np.arange(len(arg1)) == arg2[arg1]
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[liste_idx1]
    array2_k = array2[liste_idx2]
    return np.hstack(
        [
            liste_idx1[:, np.newaxis],
            liste_idx2[:, np.newaxis],
            array1_k[:, np.newaxis],
            array2_k[:, np.newaxis],
            (array1_k - array2_k)[:, np.newaxis],
        ]
    )


def rolling_iq(array, window=1, min_periods=1):
    """
    Perform a rolling IQ statistic in a fixed window.

    Parameters
    ----------
    array : array_like
        The vector to investigate.
    window : int
        The window used for the rolling statistic.
    min_periods: int
        Computation of the statistics up to the min_periods border value

    Returns
    -------
    rolling_Q1 : array_like
        The rolling 25th percentile.
    rolling_Q3 : array_like
        The rolling 75th percentile.
    rolling_IQ : array_like
        The rolling IQ (Q3-Q1).
    """
    roll_Q1 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.25)
    )
    roll_Q3 = np.ravel(
        pd.DataFrame(array).rolling(window, min_periods=min_periods, center=True).quantile(0.75)
    )
    roll_IQ = roll_Q3 - roll_Q1
    return roll_Q1, roll_Q3, roll_IQ
