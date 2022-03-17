# def suppress_low_snr_spectra(files_to_process, snr_cutoff=100, suppress=True):
#     for j in files_to_process:
#         file = pd.read_pickle(j)
#         if "parameters" not in file.keys():
#             if file["SNR_5500"] < snr_cutoff:
#                 print("File deleted : %s " % (j))
#                 if suppress:
#                     os.system("rm " + j)
#                 else:
#                     new_name = "rassine_" + j.split("_")[1]
#                     os.system("mv " + j + " " + new_name)
#         else:
#             if file["parameters"]["SNR_5500"] < snr_cutoff:
#                 print("File deleted : %s " % (j))
#                 if suppress:
#                     os.system("rm " + j)
#                 else:
#                     new_name = "rassine_" + j.split("_")[1]
#                     os.system("mv " + j + " " + new_name)


# def suppress_ccd_gap(files_to_process, continuum="linear"):
#     """Fill the gap between the ccd of HARPS s1d"""
#     for i, j in enumerate(files_to_process):
#         print("Modification of file (%.0f/%.0f) : %s" % (i + 1, len(files_to_process), j))
#         file = pd.read_pickle(j)
#         conti = file["matching_anchors"]["continuum_" + continuum]
#         flux_norm = file["flux"] / conti
#         cluster = grouping(flux_norm, 0.001, 0)[-1]
#         cluster = cluster[cluster[:, 2].argmax(), :]
#         left = cluster[0] - 10
#         right = cluster[1] + 10
#         flux_norm[int(left) : int(right) + 1] = 1
#         new_flux = smooth(flux_norm, box_pts=6, shape="gaussian")
#         flux_norm[left - 6 : right + 7] = new_flux[left - 6 : right + 7]
#         file["flux"] = flux_norm * conti
#         save_pickle(j, file)


# def plot_debug(grid, spectre, wave, flux):
#     plt.figure()
#     plt.plot(grid, spectre)
#     plt.scatter(wave, flux, color="k")
#     plt.show(block=False)
#     input("blabla")
