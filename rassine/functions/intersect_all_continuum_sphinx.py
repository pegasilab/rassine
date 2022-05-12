import glob as glob
import multiprocessing as multicpu
import os
import time
from itertools import repeat
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pylab as plt
import numpy as np
from colorama import Fore
from matplotlib.widgets import Button, Slider

from rassine.functions.misc import make_sound, sphinx

from ..analysis import grouping
from ..io import open_pickle, save_pickle


def import_files_mcpu_wrapper(args):
    return import_files_mcpu(*args)


def import_files_mcpu(file_liste, kind):
    file_liste = file_liste.tolist()
    # print(file_liste)
    sub = []
    snr = []
    for j in file_liste:
        file = open_pickle(j)
        snr.append(file["parameters"]["SNR_5500"])
        sub.append(file["output"][kind])
    return sub, snr


def intersect_all_continuum_sphinx(
    names_: Sequence[str],
    master_spectrum: Optional[str] = None,
    copies_master: int = 0,
    kind: str = "anchor_index",
    nthreads: int = 6,
    fraction: float = 0.2,
    threshold: float = 0.66,
    tolerance: float = 0.5,
    add_new: bool = True,
    feedback: bool = True,
):
    """
    Search for the intersection of all the anchors points in a list of filename and update the selection of anchor points in the same files.

    For each anchor point the fraction of the closest distance to a neighborhood is used.
    Anchor point under the threshold are removed (1 = most deleted, 0 = most kept).
    Possible to use multiprocessing with nthreads cpu.
    If you want to fix the anchors points, enter a master spectrum path and the number of copies you want of it.

    Args:
        names_: List of RASSINE files.
        master_spectrum: Name of the RASSINE master spectrum file.
        copies_master: Number of copy of the master. If 0 value is specified, copies_master is set to 2*N with N the number of RASSINE files.
        kind: Entry kw of the vector to select in the RASSINE files
        nthreads: Number of threads for multiprocessing
        fraction: Parameter of the model between 0 and 1
        treshold: Parameter of the model between 0 and 1
        tolerance: Parameter of the model between 0 and 1
        add_new: Add anchor points that were not detected.
        feedback: Activate the GUI matplotlib interface
    """

    print("Loading of the files, wait ... \n")
    names = np.sort(names_)
    sub_dico = "output"

    directory, dustbin = os.path.split(names[0])
    directory = "/".join(directory.split("/")[0:-1]) + "/MASTER/"

    previous_files = glob.glob(directory + "Master_tool*.p")
    for file_to_delete in previous_files:
        os.system("rm " + file_to_delete)

    save = []
    snr = []

    if nthreads >= multicpu.cpu_count():
        print(
            f"Your number of cpu ({multicpu.cpu_count()}) is smaller than the number your entered ({nthreads}), enter a smaller value please"
        )
    else:
        if (
            nthreads == 1
        ):  # was os.getcwd()=='/Users/cretignier/Documents/Python': #to reestablish Rassine on my computer only
            product = [
                import_files_mcpu(names, kind)
            ]  # multiprocess not work for some reason, go back to classical loop
        else:
            chunks = np.array_split(names, nthreads)
            pool = multicpu.Pool(processes=nthreads)
            product = pool.map(import_files_mcpu_wrapper, zip(chunks, repeat(kind)))

        for j in range(len(product)):
            save = save + product[j][0]
            snr = snr + product[j][1]

    snr = np.array(snr)

    if master_spectrum is not None:
        if copies_master == 0:
            print(
                "You have to specify the number of copy of the master file you want as argument."
            )
            copies_master = 2 * len(names)
            print(
                Fore.YELLOW
                + f"[WARNING] Default value of master copies fixed at {copies_master:.0f}."
                + Fore.WHITE
            )
        file = open_pickle(Path(master_spectrum))
        for j in range(copies_master):
            names = np.hstack([names, master_spectrum])
            save.append(file[sub_dico][kind])

    sum_mask = []
    all_idx = np.hstack(save)

    # names = np.array(names)[np.array(snr).argsort()[::-1]]
    save = np.array(save)  # [np.array(snr).argsort()[::-1]]
    # snr  = np.array(snr)[np.array(snr).argsort()[::-1]]

    print("Computation of the intersection of the anchors points, wait ... \n")
    for j in range(len(names)):
        # plt.scatter(save[j],j*np.ones(len(save[j])))
        diff = np.min([np.diff(save[j][1:]), np.diff(save[j][0:-1])], axis=0)
        diff = np.array([diff[0]] + list(diff) + [diff[-1]])
        diff = diff * fraction
        diff = diff.astype("int")
        mask = np.zeros(len(open_pickle(names[0])["wave"]))
        new = []
        for k in range(len(save[j])):
            new.append(save[j][k] + np.arange(-diff[k], diff[k]))
        new = np.unique(np.hstack(new))
        new = new[(new > 0) & (new < len(mask))]
        mask[new.astype("int")] = 1
        sum_mask.append(mask)
    sum_mask = np.array(sum_mask)
    sum_mask_vert = np.sum(sum_mask, axis=0)

    strat = np.linspace(int(sum_mask_vert.min()), int(sum_mask_vert.max()), 10).astype("int")
    strat = strat[strat != strat[1]]  # suppress the first level

    for j in range(len(strat) - 1)[::-1]:
        sum_mask_vert[(sum_mask_vert >= strat[j]) & (sum_mask_vert < strat[j + 1])] = strat[j]
    for j in range(len(strat[0:-1]))[::-1]:
        sum_mask_vert[sum_mask_vert == strat[0:-1][j]] = strat[j + 1]

    # sum_mask_vert -= np.diff(strat)[0]
    sum_mask_vert[sum_mask_vert == np.unique(sum_mask_vert)[0]] = 0

    mask_vert2 = sum_mask_vert.copy()
    mask_vert2[0] = 0
    mask_vert2[-1] = 0

    for j in range(len(mask_vert2) - 2):
        j += 1
        if (mask_vert2[j] > mask_vert2[j - 1]) | (
            mask_vert2[j] > mask_vert2[j + 1]
        ):  # supression of delta peak (not possible because all the peaks cannot be situated at the exact same wavelength), allow the use of grouping function here after
            mask_vert2[j] = np.max([mask_vert2[j - 1], mask_vert2[j + 1]])

    val, border = grouping(mask_vert2, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    null = np.where(border[:, -1] == 0)[0]

    area = []
    small_length = []
    small_center = []
    center2 = []
    for j in range(len(null) - 1):
        area.append(
            np.sum(border[null[j] + 1 : null[j + 1], 2] * border[null[j] + 1 : null[j + 1], 3])
        )
        peak = np.where(
            border[null[j] + 1 : null[j + 1], 3] == (border[null[j] + 1 : null[j + 1], 3].max())
        )[0]
        peak = peak[border[null[j] + 1 : null[j + 1], 2][peak].argmax()]
        small_length.append(border[null[j] + 1 : null[j + 1], 2][peak])
        small_center.append(border[null[j] + 1 : null[j + 1], 0][peak] + small_length[-1] / 2)
        center2.append(
            np.median(
                all_idx[(all_idx > border[null[j] + 1, 1]) & (all_idx < border[null[j + 1], 0])]
            )
        )
    center2 = np.round(center2, 0).astype("int")

    left = border[null, 1][0:-1]
    right = border[null, 0][1:]

    center2 = []
    for i, j in zip(left, right):
        center2.append(np.median(all_idx[(all_idx >= i) & (all_idx <= j)]))
    center2 = np.round(center2, 0).astype("int")
    center2[center2 < 0] = 0

    large_length = right - left
    large_center = left + large_length / 2

    center = np.mean(np.array([[small_center], [large_center]]), axis=0)[0]
    windows = np.mean(np.array([[small_length], [large_length]]), axis=0)[0]
    height = area / windows

    center = np.round(center, 0).astype("int")
    windows = np.round(windows / 2, 0).astype("int")
    height = np.round(height, 0).astype("int")

    center = center2

    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)
    plt.subplot(2, 1, 1)
    file_to_plot = open_pickle(names[snr.argsort()[-1]])
    plt.plot(
        file_to_plot["wave"],
        file_to_plot["flux"] / file_to_plot[sub_dico]["continuum_linear"],
        color="k",
    )
    ax = plt.gca()
    plt.ylabel("Flux normalized", fontsize=14)
    plt.title("Selection of the clusters", fontsize=14)

    plt.subplot(2, 1, 2, sharex=ax)
    for i, j in enumerate(names):
        file_to_read = open_pickle(j)
        plt.scatter(
            file_to_read[sub_dico]["anchor_wave"],
            i * np.ones(len(file_to_read[sub_dico]["anchor_wave"])),
            alpha=0.5,
        )
    plt.plot(file_to_plot["wave"], sum_mask_vert, color="g")
    plt.axhline(y=(len(names) * 1.02), color="r")
    plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
    plt.ylabel("N° of the spectrum", fontsize=14)

    stack_vert3 = np.zeros(len(sum_mask_vert))

    stack_vert = np.zeros(len(sum_mask_vert))
    for j in range(len(save)):
        stack_vert[save[j]] += 1

    for j in range(len(center)):
        if center[j] != np.nan:
            stack_vert3[center[j] - windows[j] : center[j] + windows[j] + 1] = height[j]

    stack_vert3[stack_vert3 < (len(names) * threshold)] = 0
    stack_vert3[stack_vert3 > (len(names))] = len(names)
    val, border = grouping(stack_vert3, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    for j in range(len(border)):
        if np.sum(stack_vert[int(border[j, 0]) : int(border[j, 1]) + 1]) < (
            border[j, 3] * tolerance
        ):
            stack_vert3[int(border[j, 0]) : int(border[j, 1]) + 1] = 0

    liste = []
    for j in range(len(stack_vert3) - 2):
        j += 1
        if (stack_vert3[j] != stack_vert3[j - 1]) | (stack_vert3[j] != stack_vert3[j + 1]):
            liste.append(j + 1)
    liste = np.array(liste)
    if len(liste) != 0:
        stack_vert3[liste] = 0

    val, border = grouping(stack_vert3, 1, 1)
    border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

    nb_cluster = np.sum(border[:, -1] != 0)

    gri = file_to_plot["wave"]
    (l1,) = plt.plot(gri, stack_vert3, color="k", lw=2)
    (l2,) = plt.plot([gri.min(), gri.max()], [len(names) * threshold] * 2, color="b")
    plt.axes((0.37, 0.57, 0.05, 0.05))
    plt.axis("off")
    l3 = plt.text(0, 0, f"Nb of cluster detected : {nb_cluster:.0f}", fontsize=14)
    axcolor = "whitesmoke"

    axtresh = plt.axes((0.1, 0.12, 0.30, 0.03), facecolor=axcolor)
    stresh = Slider(axtresh, "Threshold", 0, 1, valinit=threshold, valstep=0.05)

    axtolerance = plt.axes((0.1, 0.05, 0.30, 0.03), facecolor=axcolor)
    stolerance = Slider(axtolerance, "Tolerance", 0, 1, valinit=tolerance, valstep=0.05)

    resetax = plt.axes((0.8, 0.05, 0.1, 0.1))
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    class Index:
        nb_clust = nb_cluster

        def update(self, val):
            tresh = stresh.val
            tol = 1 - stolerance.val

            stack_vert3 = np.zeros(len(sum_mask_vert))

            stack_vert = np.zeros(len(sum_mask_vert))
            for j in range(len(save)):
                stack_vert[save[j]] += 1

            for j in range(len(center)):
                if center[j] != np.nan:
                    stack_vert3[center[j] - windows[j] : center[j] + windows[j] + 1] = height[j]

            stack_vert3[stack_vert3 < (len(names) * tresh)] = 0
            stack_vert3[stack_vert3 > (len(names))] = len(names)
            val, border = grouping(stack_vert3, 1, 1)
            border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

            for j in range(len(border)):
                if np.sum(stack_vert[int(border[j, 0]) : int(border[j, 1]) + 1]) < (
                    border[j, 3] * tol
                ):
                    stack_vert3[int(border[j, 0]) : int(border[j, 1]) + 1] = 0

            liste = []
            for j in range(len(stack_vert3) - 2):
                j += 1
                if (stack_vert3[j] != stack_vert3[j - 1]) | (stack_vert3[j] != stack_vert3[j + 1]):
                    liste.append(j + 1)
            liste = np.array(liste)
            if len(liste) != 0:
                stack_vert3[liste] = 0

            val, border = grouping(stack_vert3, 1, 1)
            border = np.hstack([border, np.array([i[0] for i in val])[:, np.newaxis]])

            nb_cluster = np.sum(border[:, -1] != 0)

            l3.set_text(f"Nb of cluster detected : {nb_cluster:.0f}")
            l1.set_ydata(stack_vert3)
            l2.set_ydata([len(names) * tresh] * 2)

            self.nb_clust = nb_cluster

            fig.canvas.draw_idle()

    callback = Index()
    stresh.on_changed(callback.update)
    stolerance.on_changed(callback.update)

    def reset(event):
        stolerance.reset()
        stresh.reset()

    button.on_clicked(reset)

    plt.show(block=False)

    if feedback:
        make_sound("The sphinx is waiting on you...")
        sphinx("Press ENTER to save the clusters locations (black curve)")

    while type(callback.nb_clust) == str:
        print("[WARNING] You have to move the sliders at least once before validating clusters")
        make_sound("Warning")
        sphinx("Press ENTER to save the clusters locations (black curve)")

    time.sleep(0.5)
    plt.close()
    time.sleep(0.5)

    threshold = stresh.val
    tolerance = stolerance.val

    # after plot

    val2, border2 = grouping(l1.get_ydata(), 1, 1)
    border2 = np.hstack([border2, np.array([i[0] for i in val2])[:, np.newaxis]])
    border2 = border2[border2[:, -1] > 0]
    curve = l1.get_ydata() > 0

    plt.close()

    output_cluster = {
        "curve": curve,
        "border": border2,
        "wave": gri,
        "threshold": f"{threshold:.2f}",
        "tolerance": f"{tolerance:.2f}",
        "fraction": f"{fraction:.2f}",
        "nb_copies_master": copies_master,
        "master_filename": master_spectrum,
    }
    # TOCHECK: why???
    # save_pickle("", output_cluster)

    tool_name = f"Master_tool_{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}.p"

    save_pickle(Path(directory + "/" + tool_name), output_cluster)
