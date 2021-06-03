# Scientific Library
from pandas import HDFStore

# Standard Library
import json
from pathlib import Path
import warnings

# Third Party
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# First Party
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


#%%

#     from tqdm.auto import tqdm

#     filename_hdf5 = "./data/out/hdf5_test.hdf5"
#     keys_hdf5 = IO_HDF5().get_keys(filename_hdf5)

#     # all_df = []
#     # for key in tqdm(keys_hdf5):
#     #     df_tmp, metadata = IO_HDF5().load(
#     #         filename=filename_hdf5, key="counts/KapK-12-1-24-Ext-1-Lib-1-Index2"
#     #     )
#     #     all_df.append(df_tmp)

#     all_df = IO_HDF5().load_multiple_keys(filename=filename_hdf5, keys=keys_hdf5)
#     df_mismatches1 = pd.concat(all_df, axis="index", ignore_index=True)

#     def concatenate(dfs, **kwargs):
#         """Concatenate while preserving categorical columns.

#         NB: We change the categories in-place for the input dataframes"""
#         from pandas.api.types import union_categoricals
#         import pandas as pd

#         # Iterate on categorical columns common to all dfs
#         for col in set.intersection(
#             *[set(df.select_dtypes(include="category").columns) for df in dfs]
#         ):
#             # Generate the union category across dfs for this column
#             uc = union_categoricals([df[col] for df in dfs])
#             # Change to union category for all dataframes
#             for df in dfs:
#                 df[col] = pd.Categorical(df[col].values, categories=uc.categories)
#         return pd.concat(dfs, **kwargs)

#     df_mismatches2 = concatenate(all_df, axis="index", ignore_index=True)

#     df_mismatches1.memory_usage(deep=True) / 1e6
#     df_mismatches1.memory_usage(deep=True).sum() / 1e6
#     df_mismatches2.memory_usage(deep=True) / 1e6
#     df_mismatches2.memory_usage(deep=True).sum() / 1e6

#     IO_HDF5().save(
#         df=df_mismatches2,
#         filename=filename_hdf5,
#         key="counts_combined",
#     )

#     df_mismatches_hdf5, metadata_hdf5 = IO_HDF5().load(
#         filename=filename_hdf5, key="counts_combined"
#     )

#     df_mismatches_hdf5.memory_usage(deep=True) / 1e6
#     df_mismatches_hdf5.memory_usage(deep=True).sum() / 1e6

#     df_mismatches_parquet = IO_Parquet().load("./data/out/parquet_test")
#     df_mismatches_parquet.memory_usage(deep=True) / 1e6
#     df_mismatches_parquet.memory_usage(deep=True).sum() / 1e6
#     df_mismatches_parquet.dtypes

#     # %timeit IO_HDF5().load(filename=filename_hdf5, key="counts_combined")
#     # %timeit IO_Parquet().load("./data/out/parquet_test")

#     filename =
#     pq.read_table("./data/out/parquet_test", filters=[('shortname', '=', "EC-Ext-14-Lib-14-Index1")])
#     pq.read_table("./data/out/parquet_test", filters=[('shortname', '=', "EC-Ext-14-Lib-14-Index1")]).to_pandas()

#     %timeit pq.read_table("./data/out/parquet_test", filters=[('shortname', '=', "EC-Ext-14-Lib-14-Index1")])
#     %timeit pq.read_table("./data/out/parquet_test/EC-Ext-14-Lib-14-Index1.parquet")


#     filename = "./data/out/pq_test.parquet"

#     metadata = cfg.to_dict()
#     table = pa.Table.from_pandas(df)
#     table = IO_Parquet()._add_metadata_to_table(table, metadata)
#     pq.write_table(table, filename, version="2.0")
#     pq.read_schema(filename)
#     pq.read_table(filename)

#     pq.read_table(filename)
#     pq.read_table(filename, read_dictionary=["tax_name"])

#     pd.read_parquet(filename).dtypes

#     df_mismatches_parquet, metadata_parquet = IO_Parquet().load(
#         "./data/out/parquet_test/Lok-75-Sample-4b-Ext-A26-Lib26A-Index1.parquet"
#     )

#     #

#     df.dtypes
#     IO_Parquet().save(
#         filename=f"./data/out/{cfg.shortname}.parquet",
#         df=df,
#         metadata=cfg.to_dict(),
#     )

#     df_mismatches_parquet, metadata_parquet = IO_Parquet().load(
#         "./data/out/KapK-198A-Ext-55-Lib-55-Index1.parquet"
#     )
#     df_mismatches_parquet.dtypes

#     # %timeit IO_HDF5().load(filename=filename_hdf5, key="counts_combined")

#     # df3 = pd.read_parquet(
#     #     path="./data/out/counts",
#     #     engine="pyarrow",
#     #     # columns=['shortname'],
#     #     filters=[("shortname", "=", "KapK-12-1-24-Ext-1-Lib-1-Index2"), ("tax_id", "=", "1")],
#     # )
#     # df3

#     pd.read_parquet(
#         path="./data/out/counts/KapK-12-1-24-Ext-1-Lib-1-Index2.parquet"
#     ).dtypes

#     filename_parquet = "./data/out/parquet_test.parquet"
#     df_mismatches, metadata = IO_Parquet().load(filename_parquet, shortname=None)

#     df3 = pd.read_parquet(
#         path="./data/out/parquet_test.parquet/",
#         engine="pyarrow",
#         # columns=['shortname'],
#         # filters=[("shortname", "=", "XXX"), ("tax_id", "=", "1")],
#     )

#     pd.read_parquet(
#         path="./data/out/parquet_test.parquet/shortname=KapK-12-1-24-Ext-1-Lib-1-Index2/207fe0f890e94c61b2602bd673b37d55.parquet"
#     ).dtypes

#     IO_Parquet().save(
#         filename="./data/out/test",
#         df=df_mismatches_hdf5,
#         metadata=cfg.to_dict(),
#         # partition_cols="shortname",
#     )

#     def f_test(x):
#         print(x)
#         return "-".join(x) + "1.parquet"

#     table = pa.Table.from_pandas(df_mismatches_hdf5)
#     table2 = IO_Parquet()._update_table_metadata(table, metadata=cfg.to_dict())
#     pq.write_to_dataset(
#         table2,
#         root_path="./data/out/test",
#         partition_cols=["shortname"],
#         partition_filename_cb=f_test,
#     )

#     pq.write_table(table2, "example.parquet", version="2.0")

#     # https://issues.apache.org/jira/browse/ARROW-6114

#     pd.read_parquet(path="./data/out/test").dtypes

#     pd.read_parquet(path="example.parquet").dtypes


# # if False:
# #     #     pass

# #     # else:

# #     df1 = df  # .iloc[:10]

# #     df2 = df1.copy(deep=True)
# #     df2["shortname"] = "XXX"
# #     categories = ["tax_id", "tax_name", "tax_rank", "strand", "shortname", "df_type"]
# #     df2 = utils.downcast_dataframe(df2, categories)

# #     df3 = pd.DataFrame.from_dict(
# #         {
# #             "a": range(2),
# #             "b": np.random.randn(2),
# #             "c": ["a", "b"],
# #             "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
# #             "df_type": ["fit_results", "fit_results"],
# #         }
# #     )

# #     df4 = pd.DataFrame.from_dict(
# #         {
# #             "a": [2, 3],
# #             "b": np.random.randn(2),
# #             "c": ["c", "d"],
# #             "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
# #             "df_type": ["fit_results", "fit_results"],
# #         }
# #     )

# #     # %time save_hdf5_test(df1, df2, df3, df4, cfg)
# #     # %timeit df_hdf5, metadata_hdf5, df2_hdf5, metadata2_hdf5 = load_hdf5_test()
# #     save_hdf5_test(df1, df2, df3, df4, cfg)
# #     df_hdf5, metadata_hdf5, df2_hdf5, metadata2_hdf5 = load_hdf5_test()

# #     # %time save_parquet_test(df1, df2, df3, df4, cfg)
# #     # %timeit df_parquet, metadata_parquet, df2_parquet, metadata2_parquet = load_parquet_test()
# #     save_parquet_test(df1, df2, df3, df4, cfg)
# #     df_parquet, metadata_parquet, df2_parquet, metadata2_parquet = load_parquet_test()

# #     # df1.to_parquet(
# #     #     path="analytics",
# #     #     engine="pyarrow",
# #     #     compression="snappy",
# #     #     partition_cols=["shortname"],
# #     # )

# #     # df2.to_parquet(
# #     #     path="analytics",
# #     #     engine="pyarrow",
# #     #     compression="snappy",
# #     #     partition_cols=["shortname"],
# #     # )

# #     # # df3 = pd.read_parquet(
# #     # #     path="analytics",
# #     # #     engine="pyarrow",
# #     # #     # columns=['shortname'],
# #     # #     filters=[("shortname", "=", "XXX"), ("tax_id", "=", "1")],
# #     # # )

# #     # # pd.read_parquet(
# #     # #     path="analytics",
# #     # #     engine="pyarrow",
# #     # #     # columns=['shortname'],
# #     # #     # filters=[('shortname', '=', 'XXX'), ('tax_id', '=', '1')]
# #     # # )

# #     # # index not important for counts
# #     # table1 = pa.Table.from_pandas(df, preserve_index=False)
# #     # table2 = pa.Table.from_pandas(df2, preserve_index=False)

# #     # # pq.write_table(table, "example.parquet", version="2.0")

# #     # # # Local dataset write
# #     # # pq.write_to_dataset(table, root_path="dataset_name", partition_cols=["shortname"])

# #     # # table3 = pq.read_table("dataset_name")
# #     # # table3.to_pandas()

# #     # #%%

# #     # # table = table3

# #     # # Path("parquet_dataset").mkdir(exist_ok=True)
# #     # # pq.write_table(table, "parquet_dataset/data1.parquet")
# #     # # pq.write_table(table2, "parquet_dataset/data2.parquet")

# #     # # dataset = ds.dataset("parquet_dataset", format="parquet")
# #     # # dataset.files

# #     # # print(dataset.schema.to_string(show_field_metadata=False))

# #     # # dataset.to_table().to_pandas()

# #     # # dataset.to_table(columns=["tax_name", "shortname"]).to_pandas()

# #     # # dataset.to_table(filter=ds.field("tax_id") == 1).to_pandas()
# #     # # ds.field("a") != 3
# #     # # ds.field("a").isin([1, 2, 3])

# #     # for table in [table1, table2]:

# #     #     pq.write_to_dataset(
# #     #         table,
# #     #         "parquet_dataset_partitioned",
# #     #         partition_cols=["df_type", "shortname"],
# #     #     )

# #     # dataset = ds.dataset(
# #     #     "parquet_dataset_partitioned",
# #     #     format="parquet",
# #     #     partitioning="hive",  # important to retreave the shortname column
# #     # )
# #     # dataset.files
# #     # dataset.to_table().to_pandas()
# #     # dataset.to_table(filter=ds.field("shortname") == "XXX").to_pandas()

# #     # table_different = pa.table(
# #     #     {
# #     #         "a": range(2),
# #     #         "b": np.random.randn(2),
# #     #         "c": ["a", "b"],
# #     #         "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
# #     #         "df_type": ["fit_results", "fit_results"],
# #     #     }
# #     # )
# #     # table_different.to_pandas()

# #     # pq.write_to_dataset(
# #     #     table_different,
# #     #     "parquet_dataset_partitioned",
# #     #     partition_cols=["df_type", "shortname"],
# #     # )

# #     # dataset = ds.dataset(
# #     #     "parquet_dataset_partitioned",
# #     #     format="parquet",
# #     #     partitioning="hive",  # important to retreave the shortname column
# #     # )
# #     # dataset.files
# #     # dataset.to_table(filter=ds.field("df_type") == "counts").to_pandas()
# #     # dataset.to_table(filter=ds.field("df_type") == "fit_results").to_pandas()
