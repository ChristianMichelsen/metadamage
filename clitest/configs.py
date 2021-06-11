from pathlib import Path

import pandas as pd
from rich import box
from rich.table import Table

from clitest import utils
from clitest.paths import storage_path_default


#%%


class Configs:
    def __init__(
        self,
        bam_files,
        ngsLCA_names,
        ngsLCA_nodes,
        ngsLCA_acc2tax,
        ngsLCA_command,
        ngsLCA_kwargs=None,
        storage_path=storage_path_default,
        max_cores=1,
        bayesian=False,
        forced=False,
    ):
        self.bam_files = sorted(bam_files)
        self.ngsLCA_names = ngsLCA_names
        self.ngsLCA_nodes = ngsLCA_nodes
        self.ngsLCA_acc2tax = ngsLCA_acc2tax
        self.ngsLCA_command = ngsLCA_command
        self.ngsLCA_kwargs = ngsLCA_kwargs
        self.storage_path = Path(storage_path)
        self.max_cores = int(max_cores)
        self.bayesian = bayesian
        self.forced = forced
        self.N_files = len(self.bam_files)
        self.N_cores = utils.get_number_of_cores_to_use(max_cores)

    def __iter__(self):
        for bam_file in self.bam_files:
            cfg = Config(self, bam_file)
            yield cfg

    def __repr__(self):
        s = f"Configs(bam_files = {self.bam_files}, storage_path = {self.storage_path})"
        return s

    def __rich_console__(self, console, options):
        yield f""
        my_table = Table(title="[b]Configuration:[/b]", box=box.MINIMAL_HEAVY_HEAD)
        my_table.add_column("Attribute", justify="left", style="cyan")
        my_table.add_column("Value", justify="center", style="magenta")

        my_table.add_row("Output directory", str(self.storage_path))
        my_table.add_row("Number of files", str(self.N_files))

        my_table.add_row("Number of cores to use", str(self.N_cores))

        my_table.add_row("Bayesian", str(self.bayesian))
        my_table.add_row("Forced", str(self.forced))

        yield my_table


#%%


class Config:
    def __init__(self, cfgs, bam_file):

        self.__dict__.update({key: val for key, val in cfgs.__dict__.items()})
        del self.bam_files
        self.bam_file = Path(bam_file)
        self.shortname = utils.extract_name(bam_file)
        self.d_ngsLCA = self.setup_ngsLCA_kwargs()

    def __repr__(self):
        s = f"Config(bam_file = {self.bam_file})"
        return s

    def to_dict(self):
        d_out = dict()
        for key, val in self.__dict__.items():
            if key == "d_ngsLCA":
                continue
            elif isinstance(val, Path):
                d_out[key] = str(val)
            else:
                d_out[key] = str(val)
        return d_out

    def set_number_of_fits(self, df_mismatch):
        self.N_fits = len(pd.unique(df_mismatch.tax_id))

    def setup_ngsLCA_kwargs(self):
        d_ngsLCA = {
            "command_path": self.ngsLCA_command,
            "storage_path": self.storage_path,
            "name": self.shortname,
            "kwargs": {
                "bam": self.bam_file,
                "names": self.ngsLCA_names,
                "nodes": self.ngsLCA_nodes,
                "acc2tax": self.ngsLCA_acc2tax,
                "outnames": self.shortname,
                "simscorelow": 0.95,
                "simscorehigh": 1.0,
            },
        }
        if self.ngsLCA_kwargs is not None:
            for key, val in self.ngsLCA_kwargs.items():
                d_ngsLCA["kwargs"][key] = val

        return d_ngsLCA

    def file_is_valid(self, name):
        return utils.file_is_valid(getattr(self, name))

    def get_folder_name(self, foldername):
        return self.storage_path / foldername

    def get_file_name(self, foldername, filename):
        parent = self.get_folder_name(foldername)
        path = parent / filename
        utils.make_sure_exists(path, check_parent=True)
        return path

    @property
    def file_lca_dir(self):
        path = self.get_folder_name(foldername="lca")
        utils.make_sure_exists(path)
        return path

    @property
    def file_lca_bdamage(self):
        return self.file_lca_dir / f"{self.shortname}.bdamage.gz"

    @property
    def file_lca(self):
        return self.file_lca_dir / f"{self.shortname}.lca"

    @property
    def file_lca_stat(self):
        return self.file_lca_dir / f"{self.shortname}.lca.stat"

    @property
    def file_mismatches_dir(self):
        path = self.get_folder_name(foldername="mismatches")
        utils.make_sure_exists(path)
        return path

    @property
    def file_mismatch(self):
        return (
            self.file_mismatches_dir
            / f"{self.shortname}.bdamage.gz.uglyprint.mismatch.txt"
        )

    @property
    def file_mismatch_stat(self):
        return (
            self.file_mismatches_dir / f"{self.shortname}.bdamage.gz.uglyprint.stat.txt"
        )

    @property
    def file_fit_result(self):
        return self.get_file_name(
            foldername="fit_results",
            filename=f"{self.shortname}.fit_result.parquet",
        )

    @property
    def file_result(self):
        return self.get_file_name(
            foldername="results",
            filename=f"{self.shortname}.result.parquet",
        )

    @property
    def file_read_ids_database(self):
        return self.get_file_name(
            foldername="read_ids_database",
            filename=f"{self.shortname}.pickledb.json",
        )

    def __rich_console__(self, console, options):
        yield f""
        my_table = Table(title="[b]cfg:[/b]", box=box.MINIMAL_HEAVY_HEAD)
        my_table.add_column("Attribute", justify="left", style="cyan")
        my_table.add_column("Value", justify="center", style="magenta")

        my_table.add_row("bam_file", str(self.bam_file))
        my_table.add_row("shortname", str(self.shortname))

        my_table.add_row("d_ngsLCA", str(self.d_ngsLCA))

        yield my_table


#%%
