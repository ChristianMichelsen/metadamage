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
from metadamage import cli_utils, counts, fits, plot, utils, results
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
            "Overall progress",
            total=N_files,
            progress_type="overall",
            # progress_type="LCA",
        )

        for filename in filenames:

            progress.advance(task_id_overall)

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

            df_fit_results = fits.get_fits(df_counts, cfg)

            df_results, df_results_LCA = results.get_results(
                cfg,
                df_counts,
                df_fit_results,
            )

            logger.debug("End of loop\n")

    # if all files were bad, raise error
    if bad_files == N_files:
        raise Exception("All files were bad!")

    result = results.Results(results_dir=Path("./data/out/results"))
    print(result.df_results)


if utils.is_ipython():

    print("Doing iPython plot")

    filenames = [
        # "./data/input/ugly/KapK_small.UglyPrint.txt"
    ]

    reload(utils)

    cfg = utils.Config(
        out_dir=Path("./data/out/"),
        # max_fits=10,
        max_fits=None,
        max_cores=-1,
        min_alignments=10,
        min_k_sum=10,
        min_N_at_each_pos=1,
        substitution_bases_forward=cli_utils.SubstitutionBases.CT.value,
        substitution_bases_reverse=cli_utils.SubstitutionBases.GA.value,
        # bayesian=True,
        bayesian=False,
        forced=False,
        version="0.0.0",
        dask_port=8787,
    )

    path = Path().cwd().parent
    os.chdir(path)

    filenames = sorted(Path("./data/").rglob("input/*.bdamage.gz"))
    cfg.add_filenames(filenames)

    filename = "./data/input/test.bdamage.gz"  # SJArg-1
    filename = "./data/input/KapK-12-1-39-Ext-19-Lib-19-Index1.col.sorted.sam.gz.bdamage.gz"  # SJArg-1
    filename = filenames[0]

    x = x

    cfg.add_filename(filename)
    df_counts = counts.load_counts(cfg)
    df_fit_results = fits.get_fits(df_counts, cfg)

    x = x

    df_results, df_results_LCA = results.get_results(
        cfg,
        df_counts,
        df_fit_results,
    )

    if False:
        # if True:
        main(filenames, cfg)

        # from metadamage import io
        # io.Parquet(
        #     "./data/out/fit_results/KapK-12-1-24-Ext-1-Lib-1-Index2.parquet"
        # ).load_metadata()

        tax_id = 1224
        tax_id = 1236
        tax_id = 9979  # KapK
        tax_id = -1

        cfg.add_filename(filename)
        df_counts = counts.load_counts(cfg)
        group = utils.get_specific_tax_id(df_counts, tax_id=tax_id)
        data = fits.group_to_numpyro_data(group, cfg)

    # x = x

    # # First Party
    # from metadamage import dashboard

    # dashboard.utils.set_custom_theme()

    # # reload(dashboard)

    # x = x

    # # reload(results)
    fit_results = results.Results(results_dir=Path("./data/out/results"))
    fit_results.df_results

    shortname = "EC-Ext-14-Lib-14-Index1"
    tax_id = 9606

    group = fit_results.get_single_count_group(
        shortname,
        tax_id,
        forward_reverse="",
    )

    fit = fit_results.get_single_fit_prediction(
        shortname,
        tax_id,
        forward_reverse="",
    )
