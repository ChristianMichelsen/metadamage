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

    filename = filenames[0]  # BPN19-AR
    filename = filenames[1]  # EC-Ext-14-
    filename = filenames[2]  # EC-Ext-A27
    filename = filenames[3]  # KapK
    filename = filenames[4]  # Lok-75
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
        tax_id = 28211
        tax_id = 8006
        tax_id = 4751
        tax_id = 469
        tax_id = 28211
        tax_id = 356
        tax_id = 286
        tax_id = 526227
        tax_id = 71240
        tax_id = 68336
        tax_id = 6072
        # tax_id = 85015 # BPN19-AR interesting
        tax_id = 237  # BPN19-AR interesting
        tax_id = -1
        group = utils.get_specific_tax_id(df_counts, tax_id=tax_id)
        data = fits.group_to_numpyro_data(group, cfg)

    from metadamage import dashboard

    dashboard.utils.set_custom_theme()

    fit_results = dashboard.fit_results.FitResults(
        folder=Path("./data/out/"),
        verbose=True,
        very_verbose=False,
    )

    fig = dashboard.figures.plot_fit_results(fit_results)

    x = x

    # fit_results.set_marker_size(marker_transformation="sqrt", marker_size_max=30)
    df = fit_results.df_fit_results

    import plotly.express as px

    def tmp_plot(x, y, x_title, y_title, range_x=(0, 1), range_y=(0, 1)):
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size="size",
            color="shortname",
            hover_name="shortname",
            # size_max=marker_size_max,
            # opacity=1,
            color_discrete_map=fit_results.d_cmap,
            custom_data=fit_results.custom_data_columns,
            range_x=range_x,
            range_y=range_y,
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
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend_title="Files",
        )

        fig.for_each_trace(
            lambda trace: dashboard.figures.set_opacity_for_trace(
                trace,
                method="sqrt",
                scale=20 / df.shortname.nunique(),
                opacity_min=0.001,
                opacity_max=0.8,
            )
        )

        fig.write_html(f"./data/out/tmp_plots/plotly__{x}__{y}.html")

        return fig

    tmp_plot(
        x="n_sigma",
        y="D_max",
        x_title="n_sigma Bayesian",
        y_title="D_max Bayesian",
        range_x=(-4, 18),
        range_y=(0, 0.6),
    )

    tmp_plot(
        x="frequentist_LR",
        y="frequentist_D_max",
        x_title="LR frequentist",
        y_title="D_max frequentist",
        range_x=(-5, 200),
        range_y=(0, 0.6),
    )

    tmp_plot(
        x="D_max",
        y="frequentist_D_max",
        x_title="D_max Bayesian",
        y_title="D_max Frequentist",
    )

    tmp_plot(
        x="q_mean",
        y="frequentist_q",
        x_title="q Bayesian",
        y_title="q Frequentist",
    )

    tmp_plot(
        x="concentration_mean",
        y="frequentist_phi",
        x_title="phi Bayesian",
        y_title="phi Frequentist",
        range_x=(2, 15_000),
        range_y=(2, 15_000),
    )

    tmp_plot(
        x="n_sigma",
        y="frequentist_LR",
        x_title="n_sigma Bayesian",
        y_title="LR frequentist",
        range_x=(-4, 18),
        range_y=(-5, 200),
    )
