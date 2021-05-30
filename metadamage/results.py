import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
import logging
from scipy.stats import betabinom as sp_betabinom


# First Party
from metadamage import counts, fits_Bayesian, fits_frequentist, io, utils
from metadamage.progressbar import progress

#%%

logger = logging.getLogger(__name__)

#%%


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def get_number_of_lines(filename):
    with open(filename, "r") as f:
        counter = 0
        for _ in f:
            counter += 1
    return counter


def read_LCA_file(filename_LCA, use_only_these_taxids=None, use_tqdm=False):

    N_lines = get_number_of_lines(filename_LCA)

    d_combined = {}

    with open(filename_LCA, "r") as f:

        task_LCA = progress.add_task(
            "task_LCA",
            progress_type="LCA",
            status="LCA    ",
            total=N_lines + 1,
        )

        for irow, row in enumerate(f):
            progress.advance(task_LCA)

            if irow == 0:
                continue

            read_id, rest = split(row.strip(), sep=":", pos=7)

            numerical, lca = split(rest, sep="\t", pos=1)

            tax_id = int(lca.split(":")[0])

            if use_only_these_taxids is not None:
                if not tax_id in use_only_these_taxids:
                    continue

            tax_name = lca.split("\t")[0].split(":")[1]
            tax_rank = lca.split("\t")[0].split(":")[2].strip('"')

            numericals = numerical.split(":")[1:]  # remove seq

            combined = (
                [read_id] + [tax_id, tax_name, tax_rank] + numericals + [lca.strip()]
            )
            d_combined[irow] = combined

    return d_combined, task_LCA


def compute_results(cfg, df_counts, df_fit_results):

    columns = [
        "read_id",
        "tax_id",
        "tax_name",
        "tax_rank",
        "seq_length_read",
        "alignments_read",
        "GC_read",
        "LCA",
    ]

    dtypes = {
        "tax_id": "uint32",
        "seq_length_read": "uint32",
        "alignments_read": "uint64",
        "GC_read": "float",
    }

    use_only_these_taxids = None
    use_only_these_taxids = set(df_fit_results.tax_id.unique())

    logger.info(f"Results: Loading LCA.")

    d_combined, task_LCA = read_LCA_file(
        cfg.filename_LCA,
        use_only_these_taxids=use_only_these_taxids,
    )

    df_LCA = pd.DataFrame.from_dict(
        d_combined,
        orient="index",
        columns=columns,
    ).astype(dtypes)

    df_grouped = df_LCA.groupby("tax_id")

    df_LCA["N_reads"] = df_grouped["read_id"].transform(len)
    # df_LCA["N_alignments"] = df_grouped["alignments_read"].transform(np.sum)

    df_LCA["seq_length_mean"] = df_grouped["seq_length_read"].transform(np.mean)
    df_LCA["seq_length_std"] = df_grouped["seq_length_read"].transform(np.std)

    df_LCA["seq_length_1%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 1)
    )
    df_LCA["seq_length_25%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 25)
    )
    df_LCA["seq_length_median"] = df_grouped["seq_length_read"].transform(np.median)
    df_LCA["seq_length_75%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 75)
    )
    df_LCA["seq_length_99%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 99)
    )

    df_LCA["gc_mean"] = df_grouped["GC_read"].transform(np.mean)
    df_LCA["gc_std"] = df_grouped["GC_read"].transform(np.std)

    df_LCA["gc_1%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 1))
    df_LCA["gc_25%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 25))
    df_LCA["gc_median"] = df_grouped["GC_read"].transform(np.median)
    df_LCA["gc_75%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 75))
    df_LCA["gc_99%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 99))

    #%%

    # merge LR and D_max into the dataframe
    # XXX IMPORTANT: Here we lose a lot of rows from the LCA
    df_LCA = pd.merge(df_LCA, df_fit_results, on=["tax_id"])

    max_pos = df_counts.position.max()
    CT_cols = [f"CT+{i}" for i in range(1, max_pos + 1)]
    GA_cols = [f"GA-{i}" for i in range(1, max_pos + 1)]

    # merge the CT transitions into the dataframe
    df_LCA = pd.merge(
        df_LCA,
        (
            df_counts.query("position > 0")
            .pivot(index="tax_id", columns="position", values="f_CT")
            .rename(columns={i + 1: col for i, col in enumerate(CT_cols)})
        ),
        on=["tax_id"],
    )

    df_LCA = pd.merge(
        df_LCA,
        (
            df_counts.query("position < 0")
            .pivot(index="tax_id", columns="position", values="f_GA")
            .sort_index(axis=1, ascending=False)
            .rename(columns={-(i + 1): col for i, col in enumerate(GA_cols)})
        ),
        on=["tax_id"],
    )

    df_LCA["shortname"] = cfg.shortname

    non_categories = ["read_id", "seq_length_read", "alignments_read", "GC_read"]
    categories = [col for col in df_LCA.columns if col not in non_categories]

    df_LCA = df_LCA.astype({cat: "category" for cat in categories})

    columns_order = [
        "tax_id",
        "tax_name",
        "tax_rank",
        "read_id",
        "seq_length_read",
        "alignments_read",
        "GC_read",
        "shortname",
        "N_reads",
        "N_alignments",
        #
        "LR",
        "D_max",
        "seq_length_mean",
        "gc_mean",
        "q",
        "A",
        "c",
        "phi",
        "rho_Ac",
        "valid",
        "asymmetry",
    ]

    columns_order += [col for col in df_fit_results.columns if not col in columns_order]

    columns_order += [
        "seq_length_std",
        "seq_length_1%",
        "seq_length_25%",
        "seq_length_median",
        "seq_length_75%",
        "seq_length_99%",
        "gc_std",
        "gc_1%",
        "gc_25%",
        "gc_median",
        "gc_75%",
        "gc_99%",
    ]
    columns_order += CT_cols + GA_cols + ["LCA"]
    df_LCA = df_LCA[columns_order]

    df = make_df_from_LCA(df_LCA)

    progress.advance(task_LCA)
    return df, df_LCA


def make_df_from_LCA(df_LCA):

    df = (
        df_LCA.groupby("tax_id")
        .first()
        .drop(columns=["read_id", "seq_length_read", "alignments_read", "GC_read"])
    )

    # get tax_id as column and not index
    df = df.reset_index()

    dtypes_non_float = {
        "tax_id": "int",
        "tax_name": "str",
        "tax_rank": "str",
        "shortname": "str",
        "LCA": "str",
        "N_reads": "int",
        "N_alignments": "int",
        "seq_length_median": "int",
        "seq_length_1%": "int",
        "seq_length_25%": "int",
        "seq_length_75%": "int",
        "seq_length_99%": "int",
    }

    for column in df.columns:
        if column not in dtypes_non_float.keys():
            dtypes_non_float[column] = "float"

    df = df.astype(dtypes_non_float)

    return df


def get_results(cfg, df_counts, df_fit_results):

    parquet_results = io.Parquet(cfg.filename_results)
    parquet_results_LCA = io.Parquet(cfg.filename_results_LCA)

    if parquet_results.exists(cfg.forced) and parquet_results_LCA.exists(cfg.forced):

        include = [
            "min_alignments",
            "min_k_sum",
            "substitution_bases_forward",
            "substitution_bases_reverse",
            "N_fits",
            "shortname",
        ]

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_results_LCA.load_metadata()

        if utils.metadata_is_similar(
            metadata_file_fit_results, metadata_cfg, include=include
        ):

            logger.info(f"Fit: Loading fits from parquet-file.")
            df_results = parquet_results.load()
            df_results_LCA = parquet_results_LCA.load()
            return df_results, df_results_LCA

    logger.info(f"Fit: Generating results and saving to file.")

    df_results, df_results_LCA = compute_results(cfg, df_counts, df_fit_results)

    parquet_results.save(df_results, metadata=cfg.to_dict())
    parquet_results_LCA.save(df_results_LCA, metadata=cfg.to_dict())

    return df_results, df_results_LCA


#%%


from pathlib import Path
from metadamage import dashboard
import plotly.express as px


def clip_df(df, column):
    if column in df.columns:
        df["_" + column] = df[column]  # save original data _column
        df[column] = np.clip(df[column], a_min=0, a_max=None)


class Results:
    def __init__(self, results_dir=Path("./data/out/results")):
        self.results_dir = Path(results_dir)
        self._load_df_results()
        self._set_cmap()
        self._set_hover_info()

    def _load_parquet_file(self, results_dir):
        df = io.Parquet(results_dir).load()
        return df

    def _load_df_results(self):
        df = self._load_parquet_file(self.results_dir)

        for column in ["LR", "forward_LR", "reverse_LR"]:
            clip_df(df, column)

        df["D_max_significance"] = df["D_max"] / df["D_max_std"]
        df["rho_Ac_abs"] = np.abs(df["rho_Ac"])

        log_columns = [
            "N_reads",
            "N_alignments",
            "LR",
            "phi",
            "k_sum_total",
            "N_sum_total",
        ]
        for column in log_columns:
            log_column = "log_" + column
            df.loc[:, log_column] = np.log10(1 + df[column])

        self.df_results = df

        self.all_tax_ids = set(self.df_results.tax_id.unique())
        self.all_tax_names = set(self.df_results.tax_name.unique())
        self.all_tax_ranks = set(self.df_results.tax_rank.unique())
        self.shortnames = list(self.df_results.shortname.unique())
        self.columns = list(self.df_results.columns)
        self.set_marker_size(variable="N_reads", function="sqrt", slider=30)

    def set_marker_size(self, variable="N_reads", function="sqrt", slider=30):

        df = self.df_results

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

        return self.df_results.query(query)

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        # blue, orange, green, red, purple, brown, pink, grey, camouflage, turquoise
        cmap = px.colors.qualitative.D3
        N_cmap = len(cmap)

        groupby = self.df_results.groupby("shortname", sort=False)

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

        columns = list(self.df_results.columns)

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
                "N_reads",
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
                "<b>Bayesian Fit Results</b>: <br>"
                "    n sigma:  %{customdata[_XXX_]:9.2f} <br>"
                "    D max:    %{customdata[_XXX_]:9.2f} <br>"
                "    q:        %{customdata[_XXX_]:9.2f} <br>"
                "    phi:      %{customdata[_XXX_]:9.3s} <br><br>"
                "<b>Counts</b>: <br>"
                "    N reads:     %{customdata[_XXX_]:6.3s} <br>"
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
                "rho_Ac",
                # Counts
                "N_reads",
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
                "    N reads:     %{customdata[_XXX_]:6.3s} <br>"
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

        self.customdata = self.df_results[self.custom_data_columns]

        self.hovertemplate_fit = (
            "Fit: <br>D(z) = %{y:.3f} ± %{error_y.array:.3f}<br>" "<extra></extra>"
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

    def load_df_counts_shortname(self, shortname, columns=None):
        intermediate_dir = self.results_dir.parent.parent / "intermediate" / "counts"
        return io.Parquet(intermediate_dir).load(shortname, columns=columns)

    def get_single_count_group(self, shortname, tax_id, forward_reverse=""):
        # query = f"shortname == '{shortname}' & tax_id == {tax_id}"
        # group = self.df_results.query(query)
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
        ds = self.df_results.query(query)
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
