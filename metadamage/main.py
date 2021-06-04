from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import reload
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd

import metadamage as meta
from metadamage.progressbar import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%


def main(cfgs):

    meta.utils.initial_print(cfgs)

    bad_files = 0

    with progress:

        task_id_overall = progress.add_task(
            "Overall progress",
            total=cfgs.N_files,
            progress_type="overall",
        )

        for cfg in cfgs:

            progress.advance(task_id_overall)

            if not cfg.file_is_valid:
                bad_files += 1
                continue

            progress.add_task(
                "task_name",
                progress_type="shortname",
                name=cfg.shortname,
            )

            df_mismatches = meta.mismatches.load(cfg)

            if not meta.utils.is_df_mismatches_accepted(cfg, df_mismatches):
                continue

            df_fit_results = meta.fits.load(cfg, df_mismatches)

            df_results, df_results_read = meta.LCA.load(
                cfg,
                df_mismatches,
                df_fit_results,
            )

            logger.debug("End of loop\n")

    # if all files were bad, raise error
    if bad_files == cfgs.N_files:
        raise Exception("All files were bad!")
