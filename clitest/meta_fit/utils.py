from dataclasses import asdict
from importlib.metadata import version
import importlib.resources as importlib_resources
import logging
from pathlib import Path
import platform
import shutil
from typing import List, Optional, Union

from click_help_colors import HelpColorsCommand, HelpColorsGroup

# import dill
import numpy as np
import pandas as pd
from psutil import cpu_count
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from scipy import stats
from scipy.stats import norm as sp_norm
from scipy.stats.distributions import chi2 as sp_chi2
import toml

from clitest.meta_fit.progressbar import console, progress


#%%


logger = logging.getLogger(__name__)


#%%


def get_number_of_cores_to_use(max_cores):

    available_cores = cpu_count(logical=True)

    if max_cores > available_cores:
        N_cores = available_cores - 1
        logger.info(
            f"'max_cores' is set to a value larger than the maximum available"
            f"so clipping to {N_cores} (available-1) cores"
        )

    elif max_cores < 0:
        N_cores = max(1, available_cores - abs(max_cores))
        logger.info(
            f"'max-cores' is set to a negative value"
            f"so using {N_cores} (available-max_cores) cores"
        )

    else:
        N_cores = max_cores

    return N_cores


#%%


class Configs:
    def __init__(
        self,
        filenames,
        out_dir=Path("./data/out/"),
        max_cores=1,
        bayesian=False,
        forced=False,
    ):
        self.filenames = remove_bad_files(filenames)
        self.out_dir = Path(out_dir)
        self.max_cores = int(max_cores)
        self.bayesian = bayesian
        self.forced = forced
        self.N_files = len(self.filenames)
        self.N_cores = get_number_of_cores_to_use(max_cores)
        self.intermediate_dir = out_dir / ".intermediate"

    def __iter__(self):
        for filename in self.filenames:
            cfg = Config(self, filename)
            yield cfg

    def __repr__(self):
        s = f"Configs(filenames = {self.filenames}, out_dir = {self.out_dir})"
        return s

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield f""
        my_table = Table(title="[b]Configuration:[/b]", box=box.MINIMAL_HEAVY_HEAD)
        my_table.add_column("Attribute", justify="left", style="cyan")
        my_table.add_column("Value", justify="center", style="magenta")

        my_table.add_row("Output directory", str(self.out_dir))
        my_table.add_row("Number of files", str(self.N_files))

        my_table.add_row("Number of cores to use", str(self.N_cores))

        my_table.add_row("Bayesian", str(self.bayesian))
        my_table.add_row("Forced", str(self.forced))

        yield my_table


class Config:
    def __init__(self, cfgs, filename):

        self.__dict__.update(
            {key: val for key, val in cfgs.__dict__.items() if key != "filenames"}
        )

        self.shortname = extract_name(filename)
        self.filename = Path(filename)
        self.out_dir = Path(self.out_dir)
        self.intermediate_dir = Path(self.intermediate_dir)

    def __repr__(self):
        s = f"Config(filename = {self.filename})"
        return s

    def to_dict(self):
        # d_out = asdict(self)
        d_out = dict(self.__dict__)
        for key, val in d_out.items():
            if isinstance(val, Path):
                d_out[key] = str(val)
        return d_out

    def set_number_of_fits(self, df_mismatches):
        self.N_fits = len(pd.unique(df_mismatches.tax_id))

    @property
    def file_is_valid(self):
        return file_is_valid(self.filename)

    @property
    def filename_mismatch(self):
        return self.filename

    @property
    def filename_mismatches_stats(self):
        return Path(str(self.filename) + ".stats")

    @property
    def filename_mismatches_parquet(self):
        if not self.intermediate_dir.exists():
            self.intermediate_dir.mkdir(parents=True)
        return self.intermediate_dir / f"{self.shortname}.mismatches.parquet"

    @property
    def filename_fit_results(self):
        if not self.intermediate_dir.exists():
            self.intermediate_dir.mkdir(parents=True)
        return self.intermediate_dir / f"{self.shortname}.fit_results.parquet"

    @property
    def filename_results(self):
        return self.out_dir / "results" / f"{self.shortname}.results.parquet"

    @property
    def filename_results_read(self):
        return self.out_dir / "reads" / f"{self.shortname}.results.read.parquet"

    @property
    def filename_LCA(self):
        return Path(str(self.filename).replace(".mismatch", ".lca"))


#%%


ACTG = ["A", "C", "G", "T"]

ref_obs_bases = []
for ref in ACTG:
    for obs in ACTG:
        ref_obs_bases.append(f"{ref}{obs}")

mismatch_suffix = ".mismatch"


def is_mismatch_file(s):
    return str(s).endswith(mismatch_suffix)


def remove_bad_files(filenames):
    """Keeps only files that ends with .mismatch.
    Returns as Paths
    """
    filenames_good = []
    for filename in filenames:
        # break
        if filename.is_dir():
            filenames_good += list(filename.glob(f"*{mismatch_suffix}"))
        elif filename.is_file() and is_mismatch_file(filename):
            filenames_good.append(filename)
    return sorted(filenames_good)


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
    return df["direction"] == "5'"


def get_forward(df):
    return df[is_forward(df)]


def get_reverse(df):
    return df[~is_forward(df)]


def get_specific_tax_id(df, tax_id):
    if tax_id == -1:
        tax_id = df.tax_id.iloc[0]
    return df.query("tax_id == @tax_id")


# def load_dill(filename):
#     with open(filename, "rb") as file:
#         return dill.load(file)


# def save_dill(filename, x):
#     init_parent_folder(filename)
#     with open(filename, "wb") as file:
#         dill.dump(x, file)


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
    from PyPDF2 import PdfFileReader

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


def is_df_mismatches_accepted(cfg, df_mismatches):
    if len(df_mismatches) > 0:
        return True

    logger.warning(
        f"{cfg.shortname}: Length of dataframe was 0. "
        "Stopping any further operations on this file."
    )
    return False


#%%


def initial_print(cfgs):

    console.print("")
    console.rule("[bold red]Initialization")
    # console.print(
    #     f"\nRunning [bold green underline]metadamage[/bold green underline] "
    #     f"on {len(filenames)} file(s) using the following configuration: \n"
    # )
    console.print(cfgs)
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
