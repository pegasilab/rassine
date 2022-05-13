from __future__ import annotations

import logging

from .config import Config, update_using_anchor_file
from .process import rassine_process

logging.getLogger().setLevel("INFO")


def cli():
    cfg: Config = Config.from_command_line_()
    if cfg.random_seed is None:
        random_seed = hash(cfg.spectrum_name.stem)
    else:
        random_seed = cfg.random_seed
    spectrum_name = cfg.spectrum_name
    if cfg.output_dir is None:
        output_dir = spectrum_name.parent
    else:
        output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.input_anchor_pickle is not None:
        assert cfg.input_anchor_pickle.exists(), "Anchor file, if provided, must exist"
        update_using_anchor_file(cfg, cfg.input_anchor_pickle)

    rassine_process(
        spectrum_name=spectrum_name,
        output_dir=output_dir,
        synthetic_spectrum=cfg.synthetic_spectrum,
        random_seed=random_seed,
        par_stretching=cfg.par_stretching,
        par_vicinity=cfg.par_vicinity,
        par_smoothing_box=cfg.par_smoothing_box,
        par_smoothing_kernel=cfg.par_smoothing_kernel,
        par_fwhm_=cfg.par_fwhm,
        CCF_mask=cfg.CCF_mask,
        mask_telluric=cfg.mask_telluric,
        par_R=cfg.par_R,
        par_Rmax=cfg.par_Rmax,
        par_reg_nu=cfg.par_reg_nu,
        denoising_dist=cfg.denoising_dist,
        count_cut_lim=cfg.count_cut_lim,
        count_out_lim=cfg.count_out_lim,
        interpol=cfg.interpolation,
        plot_end=cfg.plot_end,
        save_last_plot=cfg.save_last_plot,
    )
