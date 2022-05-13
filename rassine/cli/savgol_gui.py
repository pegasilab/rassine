import argparse
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pylab as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from ..analysis import rolling_iq
from ..io import open_pickle, save_pickle
from ..misc import smooth


def my_input(text):
    return input(text)


def sphinx(sentence, rep=None, s2=""):
    answer = "-99.9"
    print(
        " ______________ \n\n --- SPHINX --- \n\n TTTTTTTTTTTTTT \n\n Question : "
        + sentence
        + "\n\n [Deafening silence ...] \n\n ______________ \n\n --- OEDIPE --- \n\n XXXXXXXXXXXXXX \n "
    )
    if rep != None:
        while answer not in rep:
            answer = my_input("Answer : " + s2)
    else:
        answer = my_input("Answer : " + s2)
    return answer


def matching_diff_continuum_sphinx(
    names_: Sequence[str],
    sub_dico="matching_anchors",
    master=None,
    savgol_window: int = 200,
    zero_point: bool = False,
):
    """
    Match the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference. The savgol window parameter can be selected by the GUI interface.

    Args:
        names_: List of RASSINE files.
        sub_dico : Name of the continuum to use. Either 'output' (RASSINE individual) or 'matching_anchors' (RASSINE time-series)
        master :  Name of the RASSINE master spectrum file.
        savgol window : Length of the window for the savgol filtering.
        zero_point : No more used ?
    """
    names = np.array(names_)

    snr = []
    for j in names:
        file = open_pickle(j)
        snr.append(file["parameters"]["SNR_5500"])

    names = np.array(names)[np.array(snr).argsort()[::-1]]
    snr = np.array(snr)[np.array(snr).argsort()[::-1]]

    if master is None:
        master = names[0]

    file_highest = open_pickle(master)
    file_highest["matching_diff"] = file_highest[sub_dico]

    save_pickle(master, file_highest)

    dx = np.diff(file_highest["wave"])[0]
    length_clip = int(100 / dx)  # smoothing on 100 \ang for the tellurics clean

    file = open_pickle(names[1])

    keys = file_highest[sub_dico].keys()
    if "continuum_cubic" in keys:
        ref_cont = "continuum_cubic"
    else:
        ref_cont = "continuum_linear"

    cont2 = file_highest["flux_used"] / file_highest[sub_dico][ref_cont]
    cont1 = file["flux_used"] / file[sub_dico][ref_cont]

    all_continuum = [
        "continuum_linear",
        "continuum_cubic",
        "continuum_linear_denoised",
        "continuum_cubic_denoised",
    ]
    continuum_to_reduce = []
    for i in all_continuum:
        if i in keys:
            continuum_to_reduce.append(i)

    diff = cont1 - cont2
    med_value = np.nanmedian(diff)
    for k in range(3):  # needed to avoid artefact induced by tellurics
        q1, q3, iq = rolling_iq(diff, window=length_clip)
        diff[(diff > q3 + 1.5 * iq) | (diff < q1 - 1.5 * iq)] = med_value
        diff[diff < q1 - 1.5 * iq] = med_value

    correction = smooth(diff, savgol_window, shape="savgol")
    correction = smooth(correction, savgol_window, shape="savgol")

    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.10, bottom=0.25, top=0.95, hspace=0.30)
    plt.title("Selection of the smoothing kernel length", fontsize=14)
    plt.plot(file["wave"], diff, color="b", alpha=0.4, label="flux difference")
    (l1,) = plt.plot(
        file["wave"], correction, color="k", label="smoothed flux difference (flux correction)"
    )
    plt.xlabel(r"Wavelength [$\AA$]", fontsize=14)
    plt.ylabel(r"$F - F_{ref}$ [normalized flux units]", fontsize=14)
    plt.legend()
    axcolor = "whitesmoke"
    axsmoothing = plt.axes((0.2, 0.1, 0.40, 0.03), facecolor=axcolor)
    ssmoothing = Slider(axsmoothing, "Kernel length", 1, 500, valinit=savgol_window, valstep=1)

    resetax = plt.axes((0.8, 0.05, 0.1, 0.1))
    button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")

    class Index:
        def update(self, val):
            smoothing = ssmoothing.val
            correction = smooth(diff, smoothing, shape="savgol")
            correction = smooth(correction, smoothing, shape="savgol")
            l1.set_ydata(correction)
            fig.canvas.draw_idle()

    callback = Index()
    ssmoothing.on_changed(callback.update)

    def reset(event):
        ssmoothing.reset()

    button.on_clicked(reset)

    plt.show(block=False)
    sphinx("Press ENTER to save the kernel length for the smoothing")
    savgol_window = ssmoothing.val
    plt.close()
    time.sleep(1)

    return master, savgol_window


def get_parser():
    res = argparse.ArgumentParser(
        description="""\
    Continuum matching tool
    
    Matches the continuum of individual spectra to a reference spectrum with a savgol filtering on the spectra difference. 
    The savogol window parameter can be selected by the GUI interface. 
    
    The input files are the ones matching the RASSINE*.p pattern in the given directory.
    """
    )

    res.add_argument(
        "--input_directory", type=Path, help="Path to the directory containing the input files"
    )
    res.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to the output text file where the savgol_window value will be written",
    )
    res.add_argument(
        "--master", type=str, default=None, help="Name of the RASSINE master spectrum file"
    )
    res.add_argument("--no-master", dest="master", action="store_const", const=None)
    res.add_argument(
        "--sub_dico",
        type=str,
        choices=["output", "matching_anchors"],
        default="matching_anchors",
        help="Name of the continuum to use. Either 'output' (RASSINE individual) or 'matching_anchors' (RASSINE time-series)",
    )
    res.add_argument(
        "--savgol_window",
        type=int,
        default=200,
        help="Length of the window for the savgol filtering",
    )
    res.add_argument("--zero_point", type=bool, default=False, help="No more used ?")
    return res


def cli():
    parser = get_parser()
    args = parser.parse_args()
    # preprocess_stack wants strings
    files = [str(path) for path in args.input_directory.glob("RASSINE*.p")]
    master, savgol_window = matching_diff_continuum_sphinx(
        files,
        sub_dico=args.sub_dico,
        master=args.master,
        savgol_window=args.savgol_window,
        zero_point=args.zero_point,
    )
    with open(args.output_file, "w") as f:
        f.write(str(savgol_window))
