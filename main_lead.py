import logging
from pathlib import Path
import shutil
from subprocess import PIPE, Popen, STDOUT
import sys

from clitest.ngsLCA_wrapper import run_ngsLCA_mismatch
from clitest.rich import console


# import clitest

# logger_name = __name__
logger_name = "clitest.main_lead"
logger = logging.getLogger(logger_name)


d_LCA = {
    "command_path": "metadamage",
    "storage_path": Path("./data"),
    "name": "test",
    "kwargs": {
        "names": "./in/names.dmp.gz",
        "nodes": "./in/names.dmp.gz",
        "acc2tax": "./in/combined_taxid_accssionNO_20200425.gz",
        "bam": "./in/subs.original.bam",
        "simscorelow": 0.95,
        "simscorehigh": 1.0,
        # "outnames": name,
    },
}

d_LCA["kwargs"]["outnames"] = d_LCA["name"]


run_ngsLCA_mismatch(d_LCA)


# run_LCA(d_LCA)
# run_mismatch(d_LCA)
# move_and_clean_LCA_files(d_LCA)
