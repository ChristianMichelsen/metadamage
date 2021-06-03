import numpy as np
import pandas as pd
import logging
from scipy.stats import betabinom as sp_betabinom
from pathlib import Path
from metadamage import dashboard
import plotly.express as px
from metadamage import io


def clip_df(df, column):
    if column in df.columns:
        df["_" + column] = df[column]  # save original data _column
        df[column] = np.clip(df[column], a_min=0, a_max=None)


def pd_wide_to_long_forward_reverse(group_wide, sep, direction):
    stub_names = ["k", "N", "f"]
    group_long = pd.wide_to_long(
        group_wide,
        stubnames=stub_names,
        i="tax_id",
        j="z",
        sep=sep,
    )[stub_names]
    group_long["direction"] = direction
    return group_long.reset_index()


def wide_to_long_df(group_wide):

    group_long_forward = pd_wide_to_long_forward_reverse(
        group_wide,
        sep="+",
        direction="Forward",
    )

    group_long_reverse = pd_wide_to_long_forward_reverse(
        group_wide,
        sep="-",
        direction="Reverse",
    )

    group_long = pd.concat([group_long_forward, group_long_reverse])

    # group_long.loc[:, ["k", "N"]] = group_long.loc[:, ["k", "N"]].astype(int)

    return group_long


class Results:
    def __init__(self, results_dir="./data/out/"):
        self.results_dir = Path(results_dir)
        self._load_df_results()
        self._set_cmap()
        self._set_hover_info()

    def _load_parquet_file(self, results_dir):
        df = io.Parquet(results_dir).load()
        return df

    def _load_df_results(self):
        df = self._load_parquet_file(self.results_dir)

        for column in ["lambda_LR", "forward_lambda_LR", "reverse_lambda_LR"]:
            clip_df(df, column)

        df["D_max_significance"] = df["D_max"] / df["D_max_std"]
        df["rho_Ac_abs"] = np.abs(df["rho_Ac"])

        log_columns = [
            "N_reads",
            # "N_alignments",
            "lambda_LR",
            "phi",
            "k_sum_total",
            "N_sum_total",
        ]
        for column in log_columns:
            log_column = "log_" + column
            df.loc[:, log_column] = np.log10(1 + df[column])

        self.df = df

        self.all_tax_ids = set(self.df.tax_id.unique())
        self.all_tax_names = set(self.df.tax_name.unique())
        self.all_tax_ranks = set(self.df.tax_rank.unique())
        self.shortnames = list(self.df.shortname.unique())
        self.columns = list(self.df.columns)
        self.set_marker_size(variable="N_reads", function="sqrt", slider=30)

    def set_marker_size(self, variable="N_reads", function="sqrt", slider=30):

        d_functions = {
            "constant": np.ones_like,
            "identity": lambda x: x,
            "sqrt": np.sqrt,
            "log10": np.log10,
        }

        self.df.loc[:, "size"] = d_functions[function](self.df[variable])

        self.max_of_size = np.max(self.df["size"])
        self.marker_size = slider

    def filter(self, filters):
        query = ""
        for column, filter in filters.items():

            if filter is None:
                continue

            elif column == "shortnames":
                query += f"(shortname in {filter}) & "

            elif column == "shortname":
                query += f"(shortname == '{filter}') & "

            elif column == "tax_id":
                query += f"(tax_id == {filter}) & "

            elif column == "tax_ids":
                query += f"(tax_id in {filter}) & "

            elif column == "tax_rank":
                query += f"(tax_rank == {filter}) & "

            elif column == "tax_ranks":
                query += f"(tax_rank in {filter}) & "

            elif column == "tax_name":
                query += f"(tax_name == {filter}) & "

            elif column == "tax_names":
                query += f"(tax_name in {filter}) & "

            else:
                low, high = filter
                if dashboard.utils.is_log_transform_column(column):
                    low = dashboard.utils.log_transform_slider(low)
                    high = dashboard.utils.log_transform_slider(high)
                query += f"({low} <= {column} <= {high}) & "

        query = query[:-2]
        # print(query)

        return self.df.query(query)

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        # blue, orange, green, red, purple, brown, pink, grey, camouflage, turquoise
        cmap = px.colors.qualitative.D3
        N_cmap = len(cmap)

        groupby = self.df.groupby("shortname", sort=False)

        symbol_counter = 0
        d_cmap = {}
        d_symbols = {}
        for i, (name, _) in enumerate(groupby):

            if (i % N_cmap) == 0 and i != 0:
                symbol_counter += 1

            d_cmap[name] = cmap[i % N_cmap]
            d_symbols[name] = symbol_counter

        self.cmap = cmap
        self.d_cmap = d_cmap
        self.d_symbols = d_symbols

        self.d_cmap_fit = {"Forward": cmap[0], "Reverse": cmap[3], "Fit": cmap[2]}

    def _set_hover_info(self):

        columns = list(self.df.columns)

        placeholder = "_XXX_"

        contains_Bayesian = any(["Bayesian" in column for column in columns])

        if contains_Bayesian:

            self.custom_data_columns = [
                "shortname",
                "tax_name",
                "tax_rank",
                "tax_id",
                # Frequentist fits
                "lambda_LR",
                "D_max",
                "D_max_std",
                "q",
                "q_std",
                "phi",
                "phi_std",
                "asymmetry",
                "rho_Ac",
                # Bayesian Fits
                "Bayesian_n_sigma",
                "Bayesian_D_max",
                "Bayesian_D_max_std",
                "Bayesian_q",
                "Bayesian_phi",
                # Counts
                "N_reads",
                # "N_alignments",
                "N_sum_total",
                "k_sum_total",
            ]

            self.hovertemplate = (
                "<b>%{customdata[_XXX_]}</b><br><br>"
                "<b>Tax</b>: <br>"
                "    Name: %{customdata[_XXX_]} <br>"
                "    Rank: %{customdata[_XXX_]} <br>"
                "    ID:   %{customdata[_XXX_]} <br><br>"
                "<b>Fit Results</b>: <br>"
                "    LR:       %{customdata[_XXX_]:9.2f} <br>"
                "    D max:    %{customdata[_XXX_]:9.2f} ± %{customdata[_XXX_]:.2f} <br>"
                "    q:        %{customdata[_XXX_]:9.2f} ± %{customdata[_XXX_]:.2f} <br>"
                "    phi:      %{customdata[_XXX_]:9.3s} ± %{customdata[_XXX_]:.3s} <br>"
                "    asymmetry:%{customdata[_XXX_]:9.3f} <br>"
                "    rho_Ac:   %{customdata[_XXX_]:9.3f} <br><br>"
                "<b>Bayesian Fit Results</b>: <br>"
                "    n sigma:  %{customdata[_XXX_]:9.2f} <br>"
                "    D max:    %{customdata[_XXX_]:9.2f} <br>"
                "    q:        %{customdata[_XXX_]:9.2f} <br>"
                "    phi:      %{customdata[_XXX_]:9.3s} <br><br>"
                "<b>Counts</b>: <br>"
                "    N reads:     %{customdata[_XXX_]:6.3s} <br>"
                # "    N alignments:%{customdata[_XXX_]:6.3s} <br>"
                "    N sum total: %{customdata[_XXX_]:6.3s} <br>"
                "    k sum total: %{customdata[_XXX_]:6.3s} <br>"
                "<extra></extra>"
            )

        else:

            self.custom_data_columns = [
                "shortname",
                "tax_name",
                "tax_rank",
                "tax_id",
                # Frequentist fits
                "lambda_LR",
                "D_max",
                "D_max_std",
                "q",
                "q_std",
                "phi",
                "phi_std",
                "asymmetry",
                "rho_Ac",
                # Counts
                "N_reads",
                # "N_alignments",
                "N_sum_total",
                "k_sum_total",
            ]

            self.hovertemplate = (
                "<b>%{customdata[_XXX_]}</b><br><br>"
                "<b>Tax</b>: <br>"
                "    Name: %{customdata[_XXX_]} <br>"
                "    Rank: %{customdata[_XXX_]} <br>"
                "    ID:   %{customdata[_XXX_]} <br><br>"
                "<b>Fit Results</b>: <br>"
                "    LR:       %{customdata[_XXX_]:9.2f} <br>"
                "    D max:    %{customdata[_XXX_]:9.2f} ± %{customdata[_XXX_]:.2f} <br>"
                "    q:        %{customdata[_XXX_]:9.2f} ± %{customdata[_XXX_]:.2f} <br>"
                "    phi:      %{customdata[_XXX_]:9.3s} ± %{customdata[_XXX_]:.3s} <br>"
                "    asymmetry:%{customdata[_XXX_]:9.3f} <br>"
                "    rho_Ac:   %{customdata[_XXX_]:9.3f} <br><br>"
                "<b>Counts</b>: <br>"
                "    N reads:     %{customdata[_XXX_]:6.3s} <br>"
                # "    N alignments:%{customdata[_XXX_]:6.3s} <br>"
                "    N sum total: %{customdata[_XXX_]:6.3s} <br>"
                "    k sum total: %{customdata[_XXX_]:6.3s} <br>"
                "<extra></extra>"
            )

        data_counter = 0
        i = 0
        while True:
            if self.hovertemplate[i : i + len(placeholder)] == placeholder:
                # break
                s_new = self.hovertemplate[:i]
                s_new += str(data_counter)
                s_new += self.hovertemplate[i + len(placeholder) :]
                self.hovertemplate = s_new
                data_counter += 1
            i += 1

            if i >= len(self.hovertemplate):
                break

        self.customdata = self.df[self.custom_data_columns]

        self.hovertemplate_fit = (
            "Fit: <br>D(z) = %{y:.3f} ± %{error_y.array:.3f}<br>" "<extra></extra>"
        )

    def parse_click_data(self, click_data, column):
        try:
            index = self.custom_data_columns.index(column)
            value = click_data["points"][0]["customdata"][index]
            return value

        except Exception as e:
            raise e

    def get_single_count_group(self, shortname, tax_id, forward_reverse=""):
        query = f"shortname == '{shortname}' & tax_id == {tax_id}"
        group_wide = self.df.query(query)
        group = wide_to_long_df(group_wide)

        if forward_reverse.lower() == "forward":
            return group.query(f"direction=='Forward'")
        elif forward_reverse.lower() == "reverse":
            return group.query(f"direction=='Reverse'")
        else:
            return group

    def get_single_fit_prediction(self, shortname, tax_id, forward_reverse=""):
        query = f"shortname == '{shortname}' & tax_id == {tax_id}"
        ds = self.df.query(query)
        if len(ds) != 1:
            raise AssertionError(f"Something wrong here, got: {ds}")

        group = self.get_single_count_group(shortname, tax_id, forward_reverse)

        if forward_reverse.lower() == "forward":
            prefix = "forward_"
        elif forward_reverse.lower() == "reverse":
            prefix = "reverse_"
        else:
            prefix = ""

        A = getattr(ds, f"{prefix}A").values
        q = getattr(ds, f"{prefix}q").values
        c = getattr(ds, f"{prefix}c").values
        phi = getattr(ds, f"{prefix}phi").values

        z = group.z.values[:15]
        N = group.N.values[:15]

        Dz = A * (1 - q) ** (np.abs(z) - 1) + c

        alpha = Dz * phi
        beta = (1 - Dz) * phi

        dist = sp_betabinom(n=N, a=alpha, b=beta)
        std = np.sqrt(dist.var()) / N

        d_out = {"mu": Dz, "std": std, "Dz": Dz, "z": z}

        return d_out


def load(results_dir=Path("./data/out/results")):
    return Results(results_dir)