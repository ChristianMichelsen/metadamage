# Standard Library
from concurrent import futures
from pathlib import Path

# Third Party
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def save_parquet_file_to_csv(file, dir_csv):
    """ Takes a single parquet file as input, loads it into pandas and saves as csv file in output directory """
    df = pq.read_table(file).to_pandas()
    file_csv = dir_csv / file.with_suffix(".csv").name
    df.to_csv(file_csv, index=False)


def run_seriel(dir_parquet, dir_csv):
    all_parquet_files = list(dir_parquet.glob("*.parquet"))
    for file in tqdm(all_parquet_files):
        save_parquet_file_to_csv(file, dir_csv)


def run_parallel(dir_parquet, dir_csv):
    all_parquet_files = list(dir_parquet.glob("*.parquet"))
    f = lambda file: save_parquet_file_to_csv(file, dir_csv)
    with futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(f, all_parquet_files), total=len(all_parquet_files)))


def convert_fit_predictions(dir_parquet, dir_csv, parallel=False):
    # make sure that output folder exists
    dir_csv.mkdir(parents=True, exist_ok=True)

    if parallel:
        run_parallel(dir_parquet, dir_csv)
    else:
        run_seriel(dir_parquet, dir_csv)
