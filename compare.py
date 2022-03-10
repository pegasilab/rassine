import pickle

import numpy as np
from recursive_diff import recursive_diff

with open(
    "/home/denis/w/rassine_original/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE_Stacked_spectrum_bin_1.2007-08-27T07:33:47.298.p",
    "rb",
) as file:
    original_data = pickle.load(file)
with open(
    "/home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE_Stacked_spectrum_bin_1.2007-08-27T07:33:47.298.p",
    "rb",
) as file:
    new_data = pickle.load(file)
print(list(recursive_diff(original_data["parameters"], new_data["parameters"])))
print(np.std(original_data["output"]["continuum_linear"] - new_data["output"]["continuum_linear"]))
print(
    np.std(
        original_data["matching_diff"]["continuum_linear"]
        - new_data["matching_diff"]["continuum_linear"]
    )
)
