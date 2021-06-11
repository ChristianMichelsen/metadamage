import logging
from pathlib import Path

from psutil import cpu_count


logger = logging.getLogger(__name__)

#%%


def get_number_of_cores_to_use(max_cores):

    available_cores = cpu_count(logical=True)

    if max_cores > available_cores:
        N_cores = available_cores - 1
        logger.debug(
            f"'max_cores' is set to a value larger than the maximum available "
            f"so clipping to {N_cores} (available-1) cores"
        )

    elif max_cores < 0:
        N_cores = max(1, available_cores - abs(max_cores))
        logger.debug(
            f"'max-cores' is set to a negative value "
            f"so using {N_cores} (available-max_cores) cores"
        )

    else:
        N_cores = max_cores

    return N_cores


def extract_name(filename, max_length=60):
    shortname = Path(filename).stem.split(".")[0]
    if len(shortname) > max_length:
        shortname = shortname[:max_length] + "..."
    logger.info(f"Running new file: {shortname}")
    return shortname


def make_sure_exists(path, check_parent=False):
    if check_parent:
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True)


def file_is_valid(filename):
    if Path(filename).exists() and Path(filename).stat().st_size > 0:
        return True


def extract_bam_files(paths):
    bam_files = []
    for path in paths:
        # break
        if path.is_file() and path.suffix == ".bam":
            bam_files.append(path)
        elif path.is_dir():
            recursive = extract_bam_files(path.glob("*.bam"))
            bam_files.extend(recursive)
    return bam_files


def init_parent_folder(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
