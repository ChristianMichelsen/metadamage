# Scientific Library
import numpy as np
import pandas as pd

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

        with about_time() as times["df_fit_predictions"]:
            self._load_df_fit_predictions()

        with about_time() as times["ranges"]:
            self._compute_ranges()

        with about_time() as times["cmap"]:
            self._set_cmap()

        with about_time() as times["hover"]:
            self._set_hover_info()

        with about_time() as times["columns"]:
            self._set_columns_scatter()

        with about_time() as times["labels"]:
            self._set_labels()

        with about_time() as times["columns_scatter_forward_reverse"]:
            self._set_columns_scatter_forward_reverse()

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

        df["N_alignments_log10"] = np.log10(df["N_alignments"])
        df["N_alignments_sqrt"] = np.sqrt(df["N_alignments"])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["N_sum_total_log10"] = np.log10(df["N_sum_total"])

        self.df_fit_results = df

        self.all_tax_ids = set(self.df_fit_results.tax_id.unique())
        self.all_tax_names = set(self.df_fit_results.tax_name.unique())
        self.all_tax_ranks = set(self.df_fit_results.tax_rank.unique())
        self.shortnames = list(self.df_fit_results.shortname.unique())
        self.columns = list(self.df_fit_results.columns)
        self.set_marker_size(marker_transformation="sqrt")

    def _load_df_fit_predictions(self):
        self.df_fit_predictions = self._load_parquet_file("fit_predictions")
        # self.df_fit_predictions = io.Parquet(self.folder / "fit_predictions").load()

    def _get_range_of_column(self, column, spacing):
        array = self.df_fit_results[column]
        array = array[np.isfinite(array) & array.notnull()]
        range_min = array.min()
        range_max = array.max()
        delta = range_max - range_min
        ranges = [range_min - delta / spacing, range_max + delta / spacing]
        return ranges

    def _compute_ranges(self, spacing=20):
        ranges = {}
        for column in self.columns:
            try:
                ranges[column] = self._get_range_of_column(column, spacing=spacing)
            except TypeError:  # skip categorical columns
                pass

        for column, range_ in ranges.items():
            if not ("_forward" in column or "_reverse" in column):
                column_forward = f"{column}_forward"
                column_reverse = f"{column}_reverse"
                if column_forward in ranges.keys() and column_reverse in ranges.keys():
                    range_forward = ranges[column_forward]
                    range_reverse = ranges[column_reverse]

                    if column == "frequentist_LR":
                        paddding = 1
                    elif column == "frequentist_D_max":
                        paddding = 0.1
                    # elif column == "noise":
                    # paddding = 1

                    if range_forward[0] < range_[0] - paddding:
                        range_forward[0] = range_[0] - paddding
                    if range_forward[1] > range_[1] + paddding:
                        range_forward[1] = range_[1] + paddding

                    if range_reverse[0] < range_[0] - paddding:
                        range_reverse[0] = range_[0] - paddding
                    if range_reverse[1] > range_[1] + paddding:
                        range_reverse[1] = range_[1] + paddding

                    ranges[column_forward] = range_forward
                    ranges[column_reverse] = range_reverse

        self.ranges = ranges

    def set_marker_size(self, marker_transformation="sqrt", marker_size_max=30):

        df = self.df_fit_results

        if isinstance(marker_transformation, list) and isinstance(
            marker_size_max, list
        ):
            if len(marker_transformation) == 0 and len(marker_size_max) == 0:
                return None
            marker_transformation = marker_transformation[0]
            marker_size_max = marker_size_max[0]

        if marker_transformation == "identity":
            df.loc[:, "size"] = df["N_alignments"]

        elif marker_transformation == "sqrt":
            df.loc[:, "size"] = np.sqrt(df["N_alignments"])

        elif marker_transformation == "log10":
            df.loc[:, "size"] = np.log10(df["N_alignments"])

        elif marker_transformation == "constant":
            df.loc[:, "size"] = np.ones_like(df["N_alignments"])

        else:
            raise AssertionError(
                f"Did not recieve proper marker_transformation: {marker_transformation}"
            )

        self.max_of_size = np.max(df["size"])
        self.marker_size_max = marker_size_max
        return None

    def filter(self, filters, df_type="df_fit_results"):
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
                if column in dashboard.utils.log_transform_columns:
                    low = dashboard.utils.log_transform_slider(low)
                    high = dashboard.utils.log_transform_slider(high)
                query += f"({low} <= {column} <= {high}) & "

        query = query[:-2]
        # print(query)

        if "fit_results" in df_type:
            return self.df_fit_results.query(query)
        # elif "counts" in df:
        # return self.df_counts.query(query)
        else:
            raise AssertionError(
                f"df_type = {df_type} not implemented yet, only 'df_fit_results'"
            )

    def get_single_count_group(self, shortname, tax_id):
        df_counts_group = self.load_df_counts_shortname(shortname)
        return df_counts_group.query(f"tax_id == {tax_id}")

    def get_single_fit_prediction(self, shortname, tax_id):
        query = f"shortname == '{shortname}' & tax_id == {tax_id}"
        return self.df_fit_predictions.query(query)

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
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

    def _set_hover_info(self):

        columns = list(self.df_fit_results.columns)

        contains_Bayesian = any(["Bayesian" in column for column in columns])
        contains_frequentist = any(["frequentist" in column for column in columns])

        if contains_Bayesian and contains_frequentist:

            self.custom_data_columns = [
                "shortname",
                "tax_name",
                "tax_rank",
                "tax_id",
                # Frequentist fits
                "frequentist_LR",
                "frequentist_D_max",
                "frequentist_D_max_std",
                "frequentist_q",
                "frequentist_phi",
                # Bayesian Fits
                "Bayesian_n_sigma",
                "Bayesian_D_max",
                "Bayesian_D_max_std",
                "Bayesian_q",
                "Bayesian_phi",
                # Counts
                "N_alignments",
                "N_sum_total",
                "y_sum_total",
            ]

            self.hovertemplate = (
                "<b>%{customdata[0]}</b><br><br>"
                "<b>Tax</b>: <br>"
                "    Name: %{customdata[1]} <br>"
                "    Rank: %{customdata[2]} <br>"
                "    ID:   %{customdata[3]} <br><br>"
                "<b>Frequentist Fit Results</b>: <br>"
                "    LR:       %{customdata[4]:9.2f} <br><br>"
                "    D max:    %{customdata[5]:9.2f} +/- %{customdata[6]:9.2f} <br>"
                "    q:        %{customdata[7]:9.2f} <br>"
                "    phi:      %{customdata[8]:9.3s} <br>"
                "<b>Bayesian Fit Results</b>: <br>"
                "    n sigma:  %{customdata[9]:9.2f} <br><br>"
                "    D max:    %{customdata[10]:9.2f} <br>"
                "    q:        %{customdata[11]:9.2f} <br>"
                "    phi:      %{customdata[12]:9.3s} <br>"
                # "    asymmetry:%{customdata[8]:9.2f} <br>"
                # "    noise:    %{customdata[9]:9.2f} <br><br>"
                "<b>Counts</b>: <br>"
                "    N alignments:%{customdata[13]:6.3s} <br>"
                "    N sum total: %{customdata[14]:6.3s} <br>"
                "    y sum total: %{customdata[15]:6.3s} <br>"
                "<extra></extra>"
            )

        else:
            raise AssertionError(f"Not yet implemented.")

        self.customdata = self.df_fit_results[self.custom_data_columns]

    def _set_columns_scatter(self):
        self.columns_scatter = [
            "frequentist_LR",
            "frequentist_D_max",
            "N_alignments_log10",
            "N_sum_total_log10",
            "frequentist_q",
            "frequentist_phi",
            # "asymmetry",
            # "normalized_noise",
        ]

    def _set_labels(self):

        labels_list = [
            r"$\large \lambda_\mathrm{LR}$",
            r"$\large D_\mathrm{max}$",
            r"$\large \log_{10} N_\mathrm{alignments}$",
            r"$\large \log_{10} N_\mathrm{sum}$",
            r"$\large \bar{q}$",
            r"$\large \bar{\phi}$",
            # r"$\large \alpha$",
            # r"$\large \mathrm{noise}$",
        ]

        iterator = zip(self.columns_scatter, labels_list)
        self.labels = {column: label for column, label in iterator}

    def _get_col_row_from_iteration(self, i, N_cols):
        col = i % N_cols
        row = (i - col) // N_cols
        col += 1
        row += 1
        return col, row

    def iterate_over_scatter_columns(self, N_cols):
        for i, column in enumerate(self.columns_scatter):
            col, row = self._get_col_row_from_iteration(i, N_cols)
            yield column, row, col

    def _set_columns_scatter_forward_reverse(self):
        self.columns_scatter_forward_reverse = {
            "frequentist_LR": r"$\large \lambda_\mathrm{LR}$",
            "frequentist_D_max": r"$\large D_\mathrm{max}$",
            "N_z1": r"$\large N_{z=1}$",
            "N_sum": r"$\large N_\mathrm{sum}$",
            "y_sum": r"$\large y_\mathrm{sum}$",
            # "normalized_noise": r"$\large \mathrm{noise}$",
        }

    def iterate_over_scatter_columns_forward_reverse(self, N_cols):
        showlegend = True
        for i, column in enumerate(self.columns_scatter_forward_reverse.keys()):
            col, row = self._get_col_row_from_iteration(i, N_cols)
            forward = f"{column}_forward"
            reverse = f"{column}_reverse"
            yield column, row, col, showlegend, forward, reverse
            showlegend = False

    def parse_click_data(self, click_data, column):
        try:
            index = self.custom_data_columns.index(column)
            value = click_data["points"][0]["customdata"][index]
            return value

        except Exception as e:
            raise e
