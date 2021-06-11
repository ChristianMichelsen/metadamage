import logging
from pathlib import Path
import shutil
from subprocess import PIPE, Popen, STDOUT

from clitest.rich import console, progress


logger_name = __name__
logger = logging.getLogger(logger_name)

# mismatch_txt_foldername = "mismatch_txt"
# lca_foldername = "lca"


def make_LCA_kwargs_into_command(cfg):
    commands = [cfg.d_ngsLCA["command_path"], "lca"]
    for key, val in cfg.d_ngsLCA["kwargs"].items():
        commands.append(f"-{key}")
        commands.append(str(val))
    return commands


def run_LCA(cfg, verbose=True):

    commands = make_LCA_kwargs_into_command(cfg)

    logger.info(f"About to run the following line with {cfg.d_ngsLCA['command_path']}:")
    logger.info(" ".join(commands))
    if verbose:
        console.print(
            f"About to run the following line with {cfg.d_ngsLCA['command_path']}:"
        )
        console.print(" ".join(commands))

    process = Popen(commands, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    # # process = Popen("./slow.sh", stdin=PIPE, stdout=PIPE, stderr=STDOUT)

    for line_raw in process.stdout:
        line = line_raw.decode("utf-8").strip()
        logger.debug(f"{line}")

    return_code = process.wait()
    logger.debug(f"{return_code=}")

    if return_code == 0:
        logger.info("ngsLCA LCA ran succesfully")
        if verbose:
            console.print("ngsLCA LCA ran succesfully")
    else:
        logger.error(f"run_LCA failed")
        raise AssertionError(f"run_LCA failed")


def run_mismatch(cfg, verbose=True):

    mismatch_kwargs = {
        "names": cfg.d_ngsLCA["kwargs"]["names"],
        "nodes": cfg.d_ngsLCA["kwargs"]["nodes"],
        "lcastat": cfg.file_lca_stat.name,
    }

    commands_mismatch = [
        cfg.d_ngsLCA["command_path"],
        "print_ugly",
        cfg.file_lca_bdamage.name,
    ]
    for key, val in mismatch_kwargs.items():
        commands_mismatch.append(f"-{key}")
        commands_mismatch.append(str(val))

    logger.info(f"About to run the following line with {cfg.d_ngsLCA['command_path']}:")
    logger.info(" ".join(commands_mismatch))
    if verbose:
        console.print(
            f"About to run the following line with {cfg.d_ngsLCA['command_path']}:"
        )
        console.print(" ".join(commands_mismatch))

    mismatch_process = Popen(commands_mismatch, stdin=PIPE, stdout=PIPE, stderr=STDOUT)

    return_code = 0
    for line_raw in mismatch_process.stdout:
        line = line_raw.decode("utf-8").strip()

        if "Could not open input BAM file" in line:
            return_code = 1

        logger.debug(f"{line}")

    if return_code == 0:
        return_code = mismatch_process.wait()

    logger.debug(f"{return_code=}")
    if return_code == 0:
        logger.info(f"print_ugly ran succesfully")
        if verbose:
            console.print(f"print_ugly ran succesfully")
    else:
        logger.error(f"run_mismatch failed")
        raise AssertionError(f"run_mismatch failed")


# delete_old_files


def make_sure_path_exists(path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def path_endswith(path, s):
    return str(path.name).endswith(s)


def move_and_clean_LCA_files(cfg, move=False):

    # lca_txt_files = list()

    suffixes = [
        ".bam.bin",
        ".lca.stat",
        ".bdamage.gz",
        ".mismatch.txt",
        ".stat.txt",
        ".lca",
        ".log",
        ".stat.txt",
        ".mismatch.txt",
    ]

    files = []
    for file in Path(".").glob(f"{cfg.shortname}.*"):
        if any((path_endswith(file, x) for x in suffixes)):
            files.append(file)

    for file in files:

        # just delete the files
        if not move:
            file.unlink()
            logger.debug(f"deleted {file}")
            continue

        if (
            path_endswith(file, ".lca")
            or path_endswith(file, ".lca.stat")
            or path_endswith(file, ".bdamage.gz")
        ):
            destination = cfg.file_lca_dir / file
            make_sure_path_exists(destination)
            shutil.move(file, destination)
            logger.debug(f"moved {file} to {destination}")

        elif path_endswith(file, ".mismatch.txt") or path_endswith(file, ".stat.txt"):
            destination = cfg.file_mismatches_dir / file
            make_sure_path_exists(destination)
            shutil.move(file, destination)
            logger.debug(f"moved {file} to {destination}")

        else:
            file.unlink()
            logger.debug(f"deleted {file}")


def run_ngsLCA_mismatch(cfg, verbose=False):

    # if file exists
    if (
        cfg.file_is_valid("file_mismatch")
        and cfg.file_is_valid("file_mismatch_stat")
        and not cfg.forced
    ):
        logger.debug(f"both {cfg.file_mismatch} and {cfg.file_mismatch_stat} exists")

    else:

        task_ngsLCA = progress.add_task(
            "task_ngsLCA",
            progress_type="unknown_total",
            status="ngsLCA ",
            total=1,
        )

        try:
            run_LCA(cfg, verbose=verbose)
            run_mismatch(cfg, verbose=verbose)
            move_and_clean_LCA_files(cfg, move=True)
            progress.advance(task_ngsLCA)
        except KeyboardInterrupt:
            logger.debug(f"Encountered KeyboardInterruption")
            move_and_clean_LCA_files(cfg)


# metadamage lca -names ./in/names.dmp.gz -nodes ./in/names.dmp.gz -acc2tax ./in/combined_taxid_accssionNO_20200425.gz -bam ./in/subs.original.bam -simscorelow 0.95 -simscorehigh 1.0 -outnames test

# metadamage print_ugly test.bdamage.gz -names ./in/names.dmp.gz -nodes ./in/names.dmp.gz -lcastat test.lca.stat
