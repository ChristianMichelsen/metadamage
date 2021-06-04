import json
from pathlib import Path
import warnings

from pandas import HDFStore
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from metadamage import utils


class Parquet:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.custom_meta_key = "metadamage"

    def __str__(self):
        return f"Parquet file: '{self.filename}'"

    def __repr__(self):
        return f"Parquet('{self.filename}')"

    def load_metadata(self):
        schema = pq.read_schema(self.filename)
        metadata_json = schema.metadata[self.custom_meta_key.encode()]
        metadata = json.loads(metadata_json)
        return metadata

    def _load_table(self, shortname=None, tax_id=None, columns=None):

        filename = self.filename
        if shortname is not None:
            filename = filename / f"{shortname}.parquet"

        if tax_id is None:
            filters = None
        else:
            filters = [("tax_id", "==", tax_id)]

        if isinstance(columns, str):
            columns = [columns]

        table = pq.read_table(filename, filters=filters, columns=columns)
        return table

    def _table_to_pandas(self, table):
        df = table.to_pandas()
        if "tax_id" in df.columns:
            df = df.astype({"tax_id": "category"})
        return df

    def load(self, shortname=None, tax_id=None, columns=None):
        table = self._load_table(shortname, tax_id=tax_id, columns=columns)
        df = self._table_to_pandas(table)
        return df

    def _add_metadata_to_table(self, table, metadata):
        if metadata is None:
            metadata = {}
        custom_meta_json = json.dumps(metadata)
        updated_metadata = {
            self.custom_meta_key.encode(): custom_meta_json.encode(),
            **table.schema.metadata,
        }
        return table.replace_schema_metadata(updated_metadata)

    def _df_to_table_with_metadata(self, df, metadata):
        table = pa.Table.from_pandas(df)
        table = self._add_metadata_to_table(table, metadata)
        return table

    def save(self, df, metadata=None):
        utils.init_parent_folder(self.filename)
        table = self._df_to_table_with_metadata(df, metadata)
        # pq.write_to_dataset(table, self.filename, partition_cols=partition_cols)
        pq.write_table(table, self.filename, version="2.0")

    # def append(self, df, metadata=None, forced=False):
    #     table = self._df_to_table_with_metadata(df, metadata)
    #     writer = pq.ParquetWriter(self.filename, table.schema)
    #     writer.write_table(table=table)

    def exists(self, forced=False):
        return self.filename.exists() and not forced


class HDF5:
    def load(self, filename, key):
        with HDFStore(filename, mode="r") as hdf:
            df = hdf.select(key)
            metadata = hdf.get_storer(key).attrs.metadata
        return df, metadata

    def load_multiple_keys(self, filename, keys):
        all_dfs = []
        with HDFStore(filename, mode="r") as hdf:
            for key in tqdm(keys):
                df_tmp = hdf.select(key)
                all_dfs.append(df_tmp)
            # metadata = hdf.get_storer(key).attrs.metadata
        return all_dfs

    def save(self, df, filename, key, metadata=None):
        utils.init_parent_folder(filename)
        if metadata is None:
            metadata = {}
        with warnings.catch_warnings():
            message = "object name is not a valid Python identifier"
            warnings.filterwarnings("ignore", message=message)
            with HDFStore(filename, mode="a") as hdf:
                hdf.append(key, df, format="table", data_columns=True)
                hdf.get_storer(key).attrs.metadata = metadata

    def get_keys(self, filename):
        with HDFStore(filename, mode="r") as hdf:
            keys = list(set(hdf.keys()))

        # remove meta keys
        keys = sorted([key for key in keys if not "/meta/" in key])
        return keys
