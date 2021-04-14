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
    filename = filenames[1]
    filename = filenames[3]
    filename = filenames[4]
    # filename = "data/input/n_sigma_test.txt"

    if False:
        # if True:
        main(filenames, cfg)

        # from metadamage import io
        # io.Parquet(
        #     "./data/out/fit_results/KapK-12-1-24-Ext-1-Lib-1-Index2.parquet"
        # ).load_metadata()

        cfg.add_filename(filename)
        df_counts = counts.load_counts(cfg)
        tax_id = 1224
        tax_id = 1236
        tax_id = 135622
        tax_id = 2742
        tax_id = 75
        tax_id = -1
        group = utils.get_specific_tax_id(df_counts, tax_id=tax_id)
        data = group_to_numpyro_data(group, cfg)

    from metadamage import dashboard

    dashboard.utils.set_custom_theme()

    fit_results = dashboard.fit_results.FitResults(
        folder=Path("./data/out/"),
        verbose=True,
        very_verbose=False,
    )

    fig = dashboard.figures.plot_fit_results(fit_results)

    x = x

    df = fit_results.df_fit_results
    # df['frequentist_D_max'] = df['frequentist_A'] + df["frequentist_c"]

    fig = px.scatter(
        df,
        x="D_max",
        y="frequentist_D_max",
        size="size",
        color="shortname",
        hover_name="shortname",
        # size_max=marker_size_max,
        # opacity=1,
        color_discrete_map=fit_results.d_cmap,
        custom_data=fit_results.custom_data_columns,
        range_x=[0, 1],
        range_y=[0, 1],
        render_mode="webgl",
        symbol="shortname",
        symbol_map=fit_results.d_symbols,
    )

    fig.update_traces(
        hovertemplate=fit_results.hovertemplate,
        marker_line_width=0,
        marker_sizeref=2.0
        * fit_results.max_of_size
        / (fit_results.marker_size_max ** 2),
    )

    fig.update_layout(
        xaxis_title=r"$\Large D_\mathrm{max}$",
        yaxis_title=r"$\Large D_\mathrm{max} frequentist$",
        legend_title="Files",
    )

    fig.for_each_trace(
        lambda trace: set_opacity_for_trace(
            trace,
            method="sqrt",
            scale=20 / df.shortname.nunique(),
            opacity_min=0.001,
            opacity_max=0.8,
        )
    )

    # fig = px.scatter(
    #     df,
    #     x="D_max",
    #     y="frequentist_D_max",
    # )