# Scientific Library
import numpy as np
import pandas as pd
from scipy.stats import betabinom as sp_betabinom

# Standard Library
from datetime import datetime
from functools import partial
from pathlib import Path

# Third Party
from about_time import about_time

# import dashboard
import dill
from joblib import Memory
import plotly.express as px

# First Party
from metadamage import dashboard, io


cachedir = "memoization"
memory = Memory(cachedir, verbose=0)

# @memory.cache


@memory.cache
def load_parquet_file_memoized(pathname, date_string):
    df = io.Parquet(pathname).load()
    return df


#%%


class FitResults:
    def __init__(self, folder, verbose=False, very_verbose=False, use_memoization=True):
        self.folder = Path(folder)
        self.verbose = verbose
        self.use_memoization = use_memoization

        times = {}

        with about_time() as times["df_fit_results"]:
            self._load_df_fit_results()

        # with about_time() as times["ranges"]:
        #     self._compute_ranges()

        with about_time() as times["cmap"]:
            self._set_cmap()

        with about_time() as times["hover"]:
            self._set_hover_info()

        if very_verbose:
            for key, val in times.items():
                print(f"\t {key}: {val.duration_human}")

    #%%

    def load_df_counts_shortname(self, shortname, columns=None):
        return io.Parquet(self.folder / "counts").load(shortname, columns=columns)

    def _load_parquet_file(self, key):
        if self.use_memoization:
            date_string = datetime.now().strftime("%Y-%d-%m")
            df = load_parquet_file_memoized(self.folder / key, date_string)
            return df
        else:
            df = io.Parquet(self.folder / key).load()
            return df

    def _load_df_fit_results(self):
        df = self._load_parquet_file("fit_results")

        df["_LR"] = df["LR"]
        df["LR"] = np.clip(df["LR"], a_min=0, a_max=None)

        df["_forward_LR"] = df["forward_LR"]
        df["forward_LR"] = np.clip(df["forward_LR"], a_min=0, a_max=None)

        df["_reverse_LR"] = df["reverse_LR"]
        df["reverse_LR"] = np.clip(df["reverse_LR"], a_min=0, a_max=None)

        df["D_max_significance"] = df["D_max"] / df["D_max_std"]
        df["rho_Ac_abs"] = np.abs(df["rho_Ac"])

        log_columns = ["LR", "phi", "N_alignments", "k_sum_total", "N_sum_total"]
        for column in log_columns:
            log_column = "log_" + column
            df.loc[:, log_column] = np.log10(1 + df[column])

        self.df_fit_results = df

        self.all_tax_ids = set(self.df_fit_results.tax_id.unique())
        self.all_tax_names = set(self.df_fit_results.tax_name.unique())
        self.all_tax_ranks = set(self.df_fit_results.tax_rank.unique())
        self.shortnames = list(self.df_fit_results.shortname.unique())
        self.columns = list(self.df_fit_results.columns)
        self.set_marker_size(variable="N_alignments", function="sqrt", slider=30)

    # def _get_range_of_column(self, column, spacing):
    #     array = self.df_fit_results[column]
    #     array = array[np.isfinite(array) & array.notnull()]
    #     range_min = array.min()
    #     range_max = array.max()
    #     delta = range_max - range_min
    #     ranges = [range_min - delta / spacing, range_max + delta / spacing]
    #     return ranges

    # def _compute_ranges(self, spacing=20):
    #     ranges = {}
    #     for column in self.columns:
    #         try:
    #             ranges[column] = self._get_range_of_column(column, spacing=spacing)
    #         except TypeError:  # skip categorical columns
    #             pass

    #     for column, range_ in ranges.items():
    #         if not ("_forward" in column or "_reverse" in column):
    #             column_forward = f"{column}_forward"
    #             column_reverse = f"{column}_reverse"
    #             if column_forward in ranges.keys() and column_reverse in ranges.keys():
    #                 range_forward = ranges[column_forward]
    #                 range_reverse = ranges[column_reverse]

    #                 if column == "LR":
    #                     paddding = 1
    #                 elif column == "D_max":
    #                     paddding = 0.1
    #                 # elif column == "noise":
    #                 # paddding = 1

    #                 if range_forward[0] < range_[0] - paddding:
    #                     range_forward[0] = range_[0] - paddding
    #                 if range_forward[1] > range_[1] + paddding:
    #                     range_forward[1] = range_[1] + paddding

    #                 if range_reverse[0] < range_[0] - paddding:
    #                     range_reverse[0] = range_[0] - paddding
    #                 if range_reverse[1] > range_[1] + paddding:
    #                     range_reverse[1] = range_[1] + paddding

    #                 ranges[column_forward] = range_forward
    #                 ranges[column_reverse] = range_reverse

    #     self.ranges = ranges

    def set_marker_size(self, variable="N_alignments", function="sqrt", slider=30):

        df = self.df_fit_results

        d_functions = {
            "constant": np.ones_like,
            "identity": lambda x: x,
            "sqrt": np.sqrt,
            "log10": np.log10,
        }

        df.loc[:, "size"] = d_functions[function](df[variable])

        self.max_of_size = np.max(df["size"])
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

        return self.df_fit_results.query(query)

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        # blue, orange, green, red, purple, brown, pink, grey, camouflage, turquoise
        cmap = px.colors.qualitative.D3
        N_cmap = len(cmap)

        groupby = self.df_fit_results.groupby("shortname", sort=False)

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

        columns = list(self.df_fit_results.columns)

        placeholder = "_XXX_"

        contains_Bayesian = any(["Bayesian" in column for column in columns])

        if contains_Bayesian:

            self.custom_data_columns = [
                "shortname",
                "tax_name",
                "tax_rank",
                "tax_id",
                # Frequentist fits
                "LR",
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
                "N_alignments",
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
                "    phi:        %{customdata[_XXX_]:9.3s} ± %{customdata[_XXX_]:.3s} <br>"
                "    asymmetry:%{customdata[_XXX_]:9.3f} <br>"
                "    rho_Ac:   %{customdata[_XXX_]:9.3f} <br><br>"
                "<b>Bayesian Fit Results</b>: <br>"
                "    n sigma:  %{customdata[_XXX_]:9.2f} <br>"
                "    D max:    %{customdata[_XXX_]:9.2f} <br>"
                "    q:        %{customdata[_XXX_]:9.2f} <br>"
                "    phi:      %{customdata[_XXX_]:9.3s} <br><br>"
                "<b>Counts</b>: <br>"
                "    N alignments:%{customdata[_XXX_]:6.3s} <br>"
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
                "LR",
                "D_max",
                "D_max_std",
                "q",
                "q_std",
                "phi",
                "phi_std",
                "asymmetry",
                # Counts
                "N_alignments",
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
                "    N alignments:%{customdata[_XXX_]:6.3s} <br>"
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

        self.customdata = self.df_fit_results[self.custom_data_columns]

        self.hovertemplate_fit = (
            "D(z) = %{y:.3f} ± %{error_y.array:.3f}<br>" "<extra></extra>"
        )

    # def _get_col_row_from_iteration(self, i, N_cols):
    #     col = i % N_cols
    #     row = (i - col) // N_cols
    #     col += 1
    #     row += 1
    #     return col, row

    def parse_click_data(self, click_data, column):
        try:
            index = self.custom_data_columns.index(column)
            value = click_data["points"][0]["customdata"][index]
            return value

        except Exception as e:
            raise e

    def get_single_count_group(self, shortname, tax_id, forward_reverse=""):
        df_counts_group = self.load_df_counts_shortname(shortname)
        group = df_counts_group.query(f"tax_id == {tax_id}").copy()
        reverse = group.position < 0
        group.loc[:, "z"] = np.abs(group["position"])
        group.loc[:, "f"] = group["f_CT"]
        group.loc[reverse, "f"] = group.loc[reverse, "f_GA"]
        group.loc[:, "direction"] = "Forward"
        group.loc[reverse, "direction"] = "Reverse"
        group.loc[:, "k"] = group["CT"]
        group.loc[reverse, "k"] = group.loc[reverse, "GA"]
        group.loc[:, "N"] = group["C"]
        group.loc[reverse, "N"] = group.loc[reverse, "G"]

        if forward_reverse.lower() == "forward":
            return group.query(f"direction=='Forward'")
        elif forward_reverse.lower() == "reverse":
            return group.query(f"direction=='Reverse'")
        else:
            return group

    def get_single_fit_prediction(self, shortname, tax_id, forward_reverse=""):
        query = f"shortname == '{shortname}' & tax_id == {tax_id}"
        ds = self.df_fit_results.query(query)
        if len(ds) != 1:
            raise AssertionError(f"Sometrhing wrong here, got: {ds}")

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


# %
