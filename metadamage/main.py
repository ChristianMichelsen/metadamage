# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Standard Library
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import reload
import logging
import os
from pathlib import Path

# Third Party
import numpyro

# First Party
from metadamage import cli_utils, counts, fits, plot, utils
from metadamage.progressbar import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%


def main(filenames, cfg):

    utils.initial_print(filenames, cfg)

    N_files = len(filenames)
    bad_files = 0

    with progress:

        task_id_overall = progress.add_task(
            f"Overall progress",
            total=N_files,
            progress_type="overall",
        )

        for filename in filenames:

            if not utils.file_is_valid(filename):
                bad_files += 1
                continue

            cfg.add_filename(filename)

            progress.add_task(
                "task_name",
                progress_type="shortname",
                name=cfg.shortname,
            )

            df_counts = counts.load_counts(cfg)
            # print(len(pd.unique(df_counts.tax_id)))
            # continue
            # group = utils.get_specific_tax_id(df_counts, tax_id=-1)  # get very first group

            if not utils.is_df_counts_accepted(df_counts, cfg):
                continue

            # fits.fit_test(df_counts, cfg)
            df_fit_results, df_fit_predictions = fits.get_fits(df_counts, cfg)

            progress.refresh()
            progress.advance(task_id_overall)
            logger.debug("End of loop\n")

    # if all files were bad, raise error
    if bad_files == N_files:
        raise Exception("All files were bad!")


if utils.is_ipython():

    print("Doing iPython plot")

    filenames = [
        # "./data/input/ugly/KapK_small.UglyPrint.txt"
    ]

    reload(utils)

    cfg = utils.Config(
        out_dir=Path("./data/out/"),
        max_fits=10,
        max_cores=-1,
        min_alignments=10,
        min_y_sum=10,
        substitution_bases_forward=cli_utils.SubstitutionBases.CT.value,
        substitution_bases_reverse=cli_utils.SubstitutionBases.GA.value,
        forced=False,
        version="0.0.0",
        dask_port=8787,
    )

    path = Path().cwd().parent
    os.chdir(path)

    filenames = sorted(Path("./data/input/").rglob("ugly/*.txt"))
    cfg.add_filenames(filenames)

    filename = filenames[0]
    # filename = filenames[1]
    filename = filenames[3]
    # filename = "data/input/n_sigma_test.txt"

    if False:
        # if True:
        main(filenames, cfg)

        # filename_parquet = cfg.filename_parquet
        filename_parquet = "./data/parquet/n_sigma_test.parquet"

        df = pd.read_parquet(filename_parquet)
        tax_id = 115547
        tax_id = 3745
        tax_id = 395312
        group = utils.get_specific_tax_id(df, tax_id)

        # from metadamage import io
        # io.Parquet(
        #     "./data/out/fit_results/KapK-12-1-24-Ext-1-Lib-1-Index2.parquet"
        # ).load_metadata()
