import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def get_number_of_lines(filename):
    with open(filename, "r") as f:
        counter = 0
        for _ in f:
            counter += 1
    return counter


def read_LCA_file(filename_LCA):

    N_lines = get_number_of_lines(filename_LCA)

    d_combined = {}

    with open(filename_LCA, "r") as f:
        for irow, row in enumerate(tqdm(f, total=N_lines, desc="Reading lines")):

            if irow == 0:
                continue

            read_id, rest = split(row.strip(), sep=":", pos=7)

            numerical, lca = split(rest, sep="\t", pos=1)
            numericals = numerical.split(":")[1:]  # remove seq

            tax_id = lca.split(":")[0]
            tax_name = lca.split("\t")[0].split(":")[1]
            tax_rank = lca.split("\t")[0].split(":")[2].strip('"')

            combined = (
                [read_id] + [tax_id, tax_name, tax_rank] + numericals + [lca.strip()]
            )
            d_combined[irow] = combined

    return d_combined


def compute_results(cfg, df_counts, df_fit_results):

    columns = [
        "read_id",
        "tax_id",
        "tax_name",
        "tax_rank",
        "seq_length_read",
        "alignments_read",
        "GC_read",
        "LCA",
    ]

    dtypes = {
        "tax_id": "uint32",
        "seq_length_read": "uint32",
        "alignments_read": "uint64",
        "GC_read": "float",
    }

    df = pd.DataFrame.from_dict(
        read_LCA_file(cfg.filename_LCA), orient="index", columns=columns
    ).astype(dtypes)

    df_grouped = df.groupby("tax_id")

    df["N_reads"] = df_grouped["read_id"].transform(len)
    df["N_alignments"] = df_grouped["alignments_read"].transform(np.sum)

    df["seq_length_mean"] = df_grouped["seq_length_read"].transform(np.mean)
    df["seq_length_std"] = df_grouped["seq_length_read"].transform(np.std)
    df["seq_length_median"] = df_grouped["seq_length_read"].transform(np.median)

    df["seq_length_1%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 1)
    )
    df["seq_length_25%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 25)
    )
    df["seq_length_75%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 75)
    )
    df["seq_length_99%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 99)
    )

    df["gc_mean"] = df_grouped["GC_read"].transform(np.mean)
    df["gc_std"] = df_grouped["GC_read"].transform(np.std)
    df["gc_median"] = df_grouped["GC_read"].transform(np.median)
    df["gc_1%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 1))
    df["gc_25%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 25))
    df["gc_75%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 75))
    df["gc_99%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 99))

    #%%

    fit_result_columns = [
        "tax_id",
        "LR",
        "D_max",
        "q",
        "phi",
        "A",
        "c",
        "rho_Ac",
        "valid",
        "asymmetry",
    ]

    # merge LR and D_max into the dataframe
    df = pd.merge(df, df_fit_results[fit_result_columns], on=["tax_id"])

    max_pos = df_counts.position.max()

    # merge the CT transitions into the dataframe
    df = pd.merge(
        df,
        (
            df_counts.query("position > 0")
            .pivot(index="tax_id", columns="position", values="f_CT")
            .rename(columns={i: f"CT+{i}" for i in range(1, max_pos + 1)})
        ),
        on=["tax_id"],
    )

    df = pd.merge(
        df,
        (
            df_counts.query("position < 0")
            .pivot(index="tax_id", columns="position", values="f_GA")
            .sort_index(axis=1, ascending=False)
            .rename(columns={-i: f"GA-{i}" for i in range(1, max_pos + 1)})
        ),
        on=["tax_id"],
    )

    non_categories = ["seq_length_read", "alignments_read", "GC_read"]
    categories = [col for col in df.columns if col not in non_categories]

    df = df.astype({cat: "category" for cat in categories})

    df_small = (
        df.groupby("tax_id")
        .first()
        .drop(columns=["read_id", "seq_length_read", "alignments_read", "GC_read"])
    )

    dtypes_small = {
        "tax_name": "str",
        "tax_rank": "str",
        "LCA": "str",
        "N_reads": "int",
        "N_alignments": "int",
        "seq_length_median": "int",
        "seq_length_1%": "int",
        "seq_length_25%": "int",
        "seq_length_75%": "int",
        "seq_length_99%": "int",
    }

    for column in df_small.columns:
        if column not in dtypes_small.keys():
            dtypes_small[column] = "float"

    df_small = df_small.astype(dtypes_small)

    columns_small_order = [
        "tax_name",
        "tax_rank",
        "N_alignments",
        "N_reads",
        "LR",
        "D_max",
        "seq_length_mean",
        "gc_mean",
        "q",
        "phi",
        "A",
        "c",
        "rho_Ac",
        "valid",
        "asymmetry",
        "seq_length_std",
        "seq_length_median",
        "seq_length_1%",
        "seq_length_25%",
        "seq_length_75%",
        "seq_length_99%",
        "gc_std",
        "gc_median",
        "gc_1%",
        "gc_25%",
        "gc_75%",
        "gc_99%",
        "CT+1",
        "CT+2",
        "CT+3",
        "CT+4",
        "CT+5",
        "GA-1",
        "GA-2",
        "GA-3",
        "GA-4",
        "GA-5",
        "LCA",
    ]

    df_small = df_small[columns_small_order]

    return df, df_small