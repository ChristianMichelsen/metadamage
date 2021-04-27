# Scientific Library
import numpy as np
import pandas as pd


def create_empty_dataframe_for_datatable(s=None):
    if s is None:
        s = "Please click any datapoint on the graph"
    return pd.DataFrame({"name": [s]})


def get_data_table_keywords(id="data_table"):

    data_table_columns_dtypes = {
        # File name
        "name": {"name": "File Name"},
        # Tax Info
        "tax_name": {"name": "Tax Name"},
        "tax_rank": {"name": "Tax Rank"},
        "tax_id": {"name": "Tax ID"},
        # Fit Results
        "n_sigma": {
            "name": "n sigma",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max": {
            "name": "D max",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "q_mean": {
            "name": "q",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "concentration_mean": {
            "name": "Concentration",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "asymmetry": {
            "name": "Assymmetry",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "normalized_noise": {
            "name": "Noise",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        # Counts
        "N_alignments": {
            "name": "N alignments",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_total": {
            "name": "N sum total",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "k_sum_total": {
            "name": "y sum total",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        # Forward & Reverse
        "n_sigma_forward": {
            "name": "n sigma, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "n_sigma_reverse": {
            "name": "n sigma, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max_forward": {
            "name": "D max, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max_reverse": {
            "name": "D max, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "N_z1_forward": {
            "name": "N z=1, forward",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_z1_reverse": {
            "name": "N z=1, reverse",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_forward": {
            "name": "N sum, forward",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_reverse": {
            "name": "N sum, reverse",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "k_sum_forward": {
            "name": "y sum, forward",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "k_sum_reverse": {
            "name": "y sum, reverse",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "normalized_noise_forward": {
            "name": "Noise, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "normalized_noise_reverse": {
            "name": "Noise, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
    }

    columns = [
        {"id": col, **dtypes} for col, dtypes in data_table_columns_dtypes.items()
    ]

    kwargs = dict(
        id=id,
        columns=columns,
        data=create_empty_dataframe_for_datatable().to_dict("records"),
        style_table={
            "overflowX": "auto",
        },
        style_data={"border": "0px"},
        style_cell={"fontFamily": "sans-serif", "fontSize": "12px"},
        # inspired by https://github.com/plotly/dash-table/issues/231
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "border": "0px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": "#F9F9F9",
                "borderTop": "1px solid black",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "#F9F9F9",
                "borderTop": "1px solid black",
                "border": "0px",
            },
        ],
    )

    return kwargs
