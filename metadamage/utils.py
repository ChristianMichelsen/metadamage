# Scientific Library
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as sp_norm
from scipy.stats.distributions import chi2 as sp_chi2

# Standard Library
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
import importlib.resources as importlib_resources
import logging
from pathlib import Path
import platform
import shutil
from typing import List, Optional, Union

# Third Party
from PyPDF2 import PdfFileReader
from click_help_colors import HelpColorsCommand, HelpColorsGroup
import dill
from joblib import Parallel
from psutil import cpu_count
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
import toml

# First Party
from metadamage.progressbar import console, progress


#%%


logger = logging.getLogger(__name__)


#%%


@dataclass
class Config:
    out_dir: Path
    #
    max_fits: Optional[int]
    max_cores: int
    # max_position: Optional[int]
    #
    min_alignments: int
    min_k_sum: int
    min_N_at_each_pos: int
    #
    substitution_bases_forward: str
    substitution_bases_reverse: str
    #
    bayesian: bool
    forced: bool
    version: str
    dask_port: int
    #
    filename: Optional[Path] = None
    shortname: Optional[str] = None

    N_filenames: Optional[int] = None
    N_fits: Optional[int] = None
    N_cores: int = field(init=False)

    def __post_init__(self):
        self._set_N_cores()

    def _set_N_cores(self):
        available_cores = cpu_count(logical=True)
        if self.max_cores > available_cores:
            self.N_cores = available_cores - 1
            logger.info(
                f"'max_cores' is set to a value larger than the maximum available"
                f"so clipping to {self.N_cores} (available-1) cores"
            )
        elif self.max_cores < 0:
            self.N_cores = available_cores - abs(self.max_cores)
            logger.info(
                f"'max-cores' is set to a negative value"
                f"so using {self.N_cores} (available-max_cores) cores"
            )
        else:
            self.N_cores = self.max_cores

    def add_filenames(self, filenames):
        self.N_filenames = len(filenames)

    def add_filename(self, filename):
        self.filename = filename
        self.shortname = extract_name(filename)

    @property
    def filename_counts(self):
        if self.shortname is None:
            raise AssertionError(
                "Shortname has to be set before filename_counts is defined: "
                "cfg.add_filename(filename) "
            )
        return self.out_dir / "counts" / f"{self.shortname}.parquet"

    @property
    def filename_fit_results(self):
        if self.shortname is None:
            raise AssertionError(
                "Shortname has to be set before filename_fit_results is defined: "
                "cfg.add_filename(filename) "
            )
        return self.out_dir / "fit_results" / f"{self.shortname}.parquet"

    @property
    def filename_fit_predictions(self):
        if self.shortname is None:
            raise AssertionError(
                "Shortname has to be set before filename_fit_predictions is defined: "
                "cfg.add_filename(filename) "
            )
        return self.out_dir / "fit_predictions" / f"{self.shortname}.parquet"

    def set_number_of_fits(self, df_counts):
        self.N_tax_ids = len(pd.unique(df_counts.tax_id))

        if self.max_fits is not None and self.max_fits > 0:
            self.N_fits = min(self.max_fits, self.N_tax_ids)

        # use all TaxIDs available
        else:
            self.N_fits = self.N_tax_ids
        logger.info(f"Setting number_of_fits to {self.N_fits}")

    def to_dict(self):
        d_out = asdict(self)
        for key, val in d_out.items():
            if isinstance(val, Path):
                d_out[key] = str(val)
        return d_out

    # def save_dict(self, dict_name):
    #     d_out = self.to_dict()
    #     with open(dict_name, "w") as f:
    #         toml.dump(d_out, f)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield f""
        my_table = Table(title="[b]Configuration:[/b]", box=box.MINIMAL_HEAVY_HEAD)
        # my_table = Table(title="Configuration:")
        my_table.add_column("Attribute", justify="left", style="cyan")
        my_table.add_column("Value", justify="center", style="magenta")

        if self.N_filenames:
            my_table.add_row("Number of files", str(self.N_filenames))

        my_table.add_row("Output directory", str(self.out_dir))

        my_table.add_row("Maximum number of fits pr. file", str(self.max_fits))
        if self.N_fits:
            my_table.add_row("Number of fits  pr. file", str(self.N_fits))

        my_table.add_row("Maximum number of cores to use", str(self.max_cores))
        if self.N_cores:
            my_table.add_row("Number of cores to use", str(self.N_cores))

        my_table.add_row("Minimum number of alignments", str(self.min_alignments))
        my_table.add_row("Minimum k sum", str(self.min_k_sum))
        my_table.add_row("Minimum N at each position", str(self.min_N_at_each_pos))

        my_table.add_row(
            "Substitution bases forward", str(self.substitution_bases_forward)
        )
        my_table.add_row(
            "Substitution bases reverse", str(self.substitution_bases_reverse)
        )

        my_table.add_row("Bayesian", str(self.bayesian))
        my_table.add_row("Forced", str(self.forced))
        my_table.add_row("Version", self.version)

        if self.filename:
            my_table.add_row("Filename", str(self.filename))
            my_table.add_row("Shortname", str(self.shortname))

        yield my_table


#%%


def find_style_file():
    with importlib_resources.path("metadamage", "style.mplstyle") as path:
        return path


def is_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


def extract_name(filename, max_length=60):
    shortname = Path(filename).stem.split(".")[0]
    if len(shortname) > max_length:
        shortname = shortname[:max_length] + "..."
    logger.info(f"Running new file: {shortname}")
    return shortname


def file_is_valid(filename):
    if Path(filename).exists() and Path(filename).stat().st_size > 0:
        return True

    exists = Path(filename).exists()
    valid_size = Path(filename).stat().st_size > 0
    logger.error(
        f"{filename} is not a valid file. "
        f"{exists=} and {valid_size=}. "
        f"Skipping for now."
    )
    return False


def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        logger.exception(f"Could not delete folder, {path}")


def init_parent_folder(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)


def is_forward(df):
    return df["strand"] == "5'"


def get_forward(df):
    return df[is_forward(df)]


def get_reverse(df):
    return df[~is_forward(df)]


def get_specific_tax_id(df, tax_id):
    if tax_id == -1:
        tax_id = df.tax_id.iloc[0]
    return df.query("tax_id == @tax_id")


def load_dill(filename):
    with open(filename, "rb") as file:
        return dill.load(file)


def save_dill(filename, x):
    init_parent_folder(filename)
    with open(filename, "wb") as file:
        dill.dump(x, file)


# def save_to_hdf5(filename, key, value):
#     with pd.HDFStore(filename, mode="a") as store:
#         store.put(key, value, data_columns=True, format="Table")


# def save_metadata_to_hdf5(filename, key, value, metadata):
#     with pd.HDFStore(filename, mode="a") as store:
#         store.get_storer(key).attrs.metadata = metadata


# def load_from_hdf5(filename, key):

#     if isinstance(key, str):
#         with pd.HDFStore(filename, mode="r") as store:
#             df = store.get(key)
#         return df

#     elif isinstance(key, (list, tuple)):
#         keys = key
#         out = []
#         with pd.HDFStore(filename, mode="r") as store:
#             for key in keys:
#                 out.append(store.get(key))
#         return out


# def load_metadata_from_hdf5(filename, key):
#     with pd.HDFStore(filename, mode="r") as store:
#         metadata = store.get_storer(key).attrs.metadata
#     return metadata


# def get_hdf5_keys(filename, ignore_subgroups=False):
#     with pd.HDFStore(filename, mode="r") as store:
#         keys = store.keys()

#     if ignore_subgroups:
#         keys = list(set([key.split("/")[1] for key in keys]))
#         return keys
#     else:
#         raise AssertionError(f"ignore_subgroups=False not implemented yet.")


#%%


def downcast_dataframe(df, categories, fully_automatic=False):

    categories = [category for category in categories if category in df.columns]

    d_categories = {category: "category" for category in categories}
    df2 = df.astype(d_categories)

    int_cols = df2.select_dtypes(include=["integer"]).columns

    if df2[int_cols].max().max() > np.iinfo("uint32").max:
        raise AssertionError("Dataframe contains too large values.")

    for col in int_cols:
        if fully_automatic:
            df2.loc[:, col] = pd.to_numeric(df2[col], downcast="integer")
        else:
            if col == "position":
                df2.loc[:, col] = df2[col].astype("int8")
            else:
                df2.loc[:, col] = df2[col].astype("uint32")

    for col in df2.select_dtypes(include=["float"]).columns:
        if fully_automatic:
            df2.loc[:, col] = pd.to_numeric(df2[col], downcast="float")
        else:
            df2.loc[:, col] = df2[col].astype("float32")

    return df2


#%%


def metadata_is_similar(metadata_file, metadata_cfg, include=None):

    # if include not defined, use all keys
    if include is None:
        # if keys are not the same, return false:
        if set(metadata_file.keys()) != set(metadata_cfg.keys()):
            return False
        include = set(metadata_file.keys())

    equals = {key: metadata_file[key] == metadata_cfg[key] for key in include}
    is_equal = all(equals.values())
    if not is_equal:
        diff = {key: val for key, val in equals.items() if val is False}
        logger.info(f"The files' metadata are not the same, differing here: {diff}")
        return False
    return True


# def metadata_is_similar(cfg, key, include=None, exclude=None):
#     """ Compares the metadata in the hdf5 file (cfg.filename_out) with that in cfg"""

#     if include is not None and exclude is not None:
#         logger.error("Cannot both include and exclude")
#         raise AssertionError(f"Cannot both include and exclude")

#     metadata_file = load_metadata_from_hdf5(filename=cfg.filename_out, key=key)
#     metadata_cfg = cfg.to_dict()

#     # # the metadata is not similar if it contains different keys
#     # if set(metadata_file.keys()) != set(metadata_cfg.keys()):
#     #     is_similar = False
#     #     logger.warning("metadata contains different keys")
#     #     return is_similar

#     if isinstance(include, (list, tuple)) and exclude is None:
#         logger.info("include is list or tuple and exclude is None")
#         # include = ['max_fits', 'max_position']
#         # exclude = None
#         is_similar = all([metadata_file[key] == metadata_cfg[key] for key in include])
#         return is_similar

#     elif isinstance(exclude, (list, tuple)) and include is None:
#         logger.info("exclude is list or tuple and include is None")
#         all_keys = metadata_file.keys()
#         similar = [
#             metadata_file[key] == metadata_cfg[key]
#             for key in all_keys
#             if key not in exclude
#         ]
#         is_similar = all(similar)
#         return is_similar

#     elif include is None and exclude is None:
#         # include = None
#         # exclude = None
#         logger.info("both include and exclude is is None")

#         if set(metadata_file.keys()) != set(metadata_cfg.keys()):
#             logger.warning("metadata contains different keys")
#             return False

#         all_keys = metadata_file.keys()
#         is_similar = all({metadata_file[key] == metadata_cfg[key] for key in all_keys})
#         return is_similar

#     else:
#         logger.warning("Did not expect to get here")
#         raise AssertionError("Did not expect to get here")


#%%


def avoid_fontconfig_warning():
    # Standard Library
    import os

    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_CTYPE"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"


def human_format(num, digits=3, mode="eng"):
    num = float(f"{num:.{digits}g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    if mode == "eng" or mode == "SI":
        translate = ["", "k", "M", "G", "T"]
    elif mode == "scientific" or mode == "latex":
        translate = ["", r"\cdot 10^3", r"\cdot 10^6", r"\cdot 10^9", r"\cdot 10^12"]
    else:
        raise AssertionError(f"'mode' has to be 'eng' or 'scientific', not {mode}.")

    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), translate[magnitude]
    )


def file_exists(filename, forced=False):
    return Path(filename).exists() and not forced


def is_pdf_valid(filename, forced=False, N_pages=None):
    try:
        if file_exists(filename, forced=forced):
            pdf_reader = PdfFileReader(filename)
            if N_pages is None:
                return True
            if N_pages == pdf_reader.numPages:
                return True
    except:
        pass
    return False


def is_macbook():
    return platform.system() == "Darwin"


#%%


# def get_percentile_as_lim(x, percentile_max=99):
#     # percentile_max = 99.5
#     percentile_min = 100 - percentile_max

#     if x.min() == 0:
#         return (0, np.percentile(x, percentile_max))
#     else:
#         return (np.percentile(x, percentile_min), np.percentile(x, percentile_max))


#%%


# def get_N_cores(cfg):

#     if cfg.N_cores > 0:
#         N_cores = cfg.N_cores
#     else:
#         N_cores = cpu_count(logical=True)
#         N_cores = N_cores - abs(cfg.N_cores)
#         if N_cores < 1:
#             N_cores = 1
#     return N_cores


#%%


def get_sorted_and_cutted_df(df, df_results, cfg):

    min_damage = cfg.min_damage if cfg.min_damage else -np.inf
    min_sigma = cfg.min_sigma if cfg.min_sigma else -np.inf
    min_alignments = cfg.min_alignments if cfg.min_alignments else -np.inf

    query = (
        f"D_max >= {min_damage} "
        + f"and n_sigma >= {min_sigma} "
        + f"and N_alignments >= {min_alignments}"
    )

    # cut away fits and TaxIDs which does not satisfy cut criteria
    df_results_cutted = df_results.query(query)
    if len(df_results_cutted) == 0:
        logger.warning(
            f"{cfg.shortname} did not have any fits that matched the requirements. "
            f"Skipping for now"
        )
        return None

    d_sort_by = {
        "alignments": "N_alignments",
        "damage": "D_max",
        "sigma": "n_sigma",
    }
    sort_by = d_sort_by[cfg.sort_by.lower()]

    # sort the TaxIDs
    df_results_cutted_ordered = df_results_cutted.sort_values(sort_by, ascending=False)

    tax_ids = df_results_cutted_ordered.index

    # get the number_of_plots in the top
    tax_ids_top = tax_ids[: cfg.number_of_plots]

    # the actual dataframe, unrelated to the fits
    df_plot = df.query("tax_id in @tax_ids_top")
    # the actual dataframe, unrelated to the fits, now sorted
    # df_plot_sorted = df_plot.sort_values(sort_by, ascending=False)
    df_plot_sorted = pd.concat(
        [df_plot.query(f"tax_id == {tax_id}") for tax_id in tax_ids_top]
    )

    return df_plot_sorted


#%%


def is_df_counts_accepted(df_counts, cfg):
    if len(df_counts) > 0:
        return True

    logger.warning(
        f"{cfg.shortname}: Length of dataframe was 0. "
        "Stopping any further operations on this file."
    )
    return False


#%%


def initial_print(filenames, cfg):

    console.print("")
    console.rule("[bold red]Initialization")
    # console.print(
    #     f"\nRunning [bold green underline]metadamage[/bold green underline] "
    #     f"on {len(filenames)} file(s) using the following configuration: \n"
    # )
    console.print(cfg)
    # console.print("")

    console.rule("[bold red]Main")
    console.print("")


#%%

# filename: Optional[str] = None
# %%


def normalize_header(cell):
    # Standard Library
    import re

    cell = re.sub(r'[-:;/\\,. \(\)#\[\]{}\$\^\n\r\xa0*><&!"\'+=%]', "_", cell)
    cell = re.sub("__+", "_", cell)
    cell = cell.strip("_")
    cell = cell.upper()
    cell = cell or "BLANK"
    return cell


#%%


def fix_latex_warnings_in_string(s):
    # https://matplotlib.org/stable/tutorials/text/usetex.html

    # fix LaTeX errors:
    replacements = [
        (r"_", r"\_"),
        (r"&", r"\&"),
        (r"#", r"\#"),
        (r"%", r"\%"),
        (r"$", r"\$"),
    ]
    # fix bad root title
    replacements.append(("root, no rank", "root"))
    for replacement in replacements:
        s = s.replace(replacement[0], replacement[1])
    return s
