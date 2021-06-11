import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


ACTG = ["A", "C", "G", "T"]

ref_obs_bases = []
for ref in ACTG:
    for obs in ACTG:
        ref_obs_bases.append(f"{ref}{obs}")

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


def is_forward(df):
    return df["direction"] == "5'"
