from .analysis import clustering, find_nearest1, grouping
from .functions.misc import (
    ccf,
    empty_ccd_gap,
    local_max,
    make_continuum,
    produce_line,
    smooth,
    sphinx,
    truncated,
)
from .io import open_pickle, save_pickle
from .math import c_lum, create_grid, doppler_r, gaussian

__all__ = [
    "ccf",
    "open_pickle",
    "create_grid",
    "try_field",
    "doppler_r",
    "gaussian",
    "smooth",
    "sphinx",
    "produce_line",
    "find_nearest1",
    "truncated",
    "empty_ccd_gap",
    "grouping",
    "local_max",
    "make_continuum",
    "c_lum",
    "save_pickle",
    "clustering",
]


def try_field(dico, field):
    try:
        a = dico[field]
        return a
    except:
        return None
