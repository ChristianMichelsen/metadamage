from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import reload
import logging
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd

from clitest import meta_fit
import clitest.meta_fit as meta
from clitest.ngsLCA_wrapper import run_ngsLCA_mismatch
from clitest.rich import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%


def main(cfgs):

    with progress:

        task_id_overall = progress.add_task(
            "Overall progress",
            total=cfgs.N_files,
            progress_type="overall",
        )

        for cfg in cfgs:

            progress.add_task(
                "cfg",
                progress_type="cfg",
                name=cfg.shortname,
            )

            run_ngsLCA_mismatch(cfg)

            df_mismatch = meta_fit.mismatch.compute(cfg)
            df_fit_results = meta_fit.fits.load(cfg, df_mismatch)
            df_results = meta_fit.results.load(cfg, df_mismatch, df_fit_results)

            progress.advance(task_id_overall)
            logger.debug("End of loop\n")

    time.sleep(1)
