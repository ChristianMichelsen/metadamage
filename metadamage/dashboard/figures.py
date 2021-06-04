from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from metadamage import dashboard


#%%


def create_empty_figure(s=None, width=None, height=None):

    if s is None:
        s = "Please select a point"

    fig = go.Figure()

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        text=s,
        font_size=20,
        showarrow=False,
    )

    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        width=width,
        height=height,
    )

    if width is not None:
        fig.update_layout(width=width)

    if height is not None:
        fig.update_layout(height=height)

    return fig


#%%


def set_opacity_for_trace(
    trace, method="sqrt", scale=3.0, opacity_min=0.001, opacity_max=0.9
):
    N = len(trace.x)
    if "lin" in method:
        opacity = 1 / N
    elif method == "sqrt":
        opacity = 1 / np.sqrt(N)
    elif method == "log":
        opacity = 1 / np.log(N)

    opacity *= scale
    opacity = max(opacity_min, min(opacity, opacity_max))

    # print(trace.name, opacity)
    trace.update(marker_opacity=opacity)


#%%


def make_figure(results, df, xaxis_column_name, yaxis_column_name, d_columns_latex):

    fig = px.scatter(
        df,
        x=xaxis_column_name,
        y=yaxis_column_name,
        size="size",
        color="shortname",
        hover_name="shortname",
        color_discrete_map=results.d_cmap,
        custom_data=results.custom_data_columns,
        render_mode="webgl",
        symbol="shortname",
        symbol_map=results.d_symbols,
    )

    # 2. * max(array of size values) / (desired maximum marker size ** 2)

    fig.update_traces(
        hovertemplate=results.hovertemplate,
        marker_line_width=0,
        marker_sizemode="area",
        marker_sizeref=2.0 * results.max_of_size / (results.marker_size ** 2),
    )

    fig.update_layout(
        xaxis_title=xaxis_column_name,
        yaxis_title=yaxis_column_name,
        showlegend=False,
        # legend_title="Files",
    )

    fig.for_each_trace(
        lambda trace: set_opacity_for_trace(
            trace,
            method="sqrt",
            scale=20 / df.shortname.nunique(),
            opacity_min=0.3,
            opacity_max=0.95,
        )
    )

    fig.update_xaxes(title=d_columns_latex[xaxis_column_name])
    fig.update_yaxes(title=d_columns_latex[yaxis_column_name])

    return fig


#%%


def plot_group(results, group, fit=None, forward_reverse=""):

    custom_data_columns = [
        "direction",
        "f",
        "k",
        "N",
    ]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "f: %{customdata[1]:8.3f} <br>"
        "k: %{customdata[2]:8d} <br>"
        "N: %{customdata[3]:8d} <br>"
        "<extra></extra>"
    )

    fig = px.scatter(
        group,
        x="z",
        y="f",
        color="direction",
        color_discrete_map=results.d_cmap_fit,
        hover_name="direction",
        custom_data=custom_data_columns,
    )

    fig.update_xaxes(
        title_text=r"$|z|$",
        title_standoff=0,
        range=[0.5, 15.5],
    )
    fig.update_yaxes(
        title=r"",
        rangemode="nonnegative",  # tozero, nonnegative
    )

    fig.update_traces(hovertemplate=hovertemplate, marker={"size": 10})

    layout = dict(
        title_text=r"",
        autosize=False,
        margin=dict(l=10, r=10, t=10, b=10),
        # hovermode="x",
        hovermode="x unified",
        hoverlabel_font_size=14,
    )

    if forward_reverse == "":
        fig.update_layout(
            **layout,
            legend=dict(
                title_text="",
                font_size=18,
                y=1.15,
                yanchor="top",
                x=0.95,
                xanchor="right",
                bordercolor="grey",
                borderwidth=1,
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
        )

        fig.add_annotation(
            # text=r"$\frac{k}{N}$",
            text=r"$k \,/ \,N$",
            x=0.01,
            xref="paper",
            xanchor="center",
            y=1.05,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
            font_size=30,
        )

    else:

        fig.update_layout(**layout, showlegend=False)

    if fit is None:
        return fig

    green_color = results.d_cmap_fit["Fit"]
    green_color_transparent = dashboard.utils.hex_to_rgb(green_color, opacity=0.1)

    # fit with errorbars
    fig.add_trace(
        go.Scatter(
            x=fit["z"],
            y=fit["mu"],
            error_y=dict(
                type="data",
                array=fit["std"],
                visible=True,
                color=green_color,
            ),
            mode="markers",
            name="Fit",
            marker_color=green_color,
            hovertemplate=results.hovertemplate_fit,
        )
    )

    # fit filled area start
    fig.add_trace(
        go.Scatter(
            x=fit["z"],
            y=fit["mu"] + fit["std"],
            mode="lines",
            line_width=0,
            showlegend=False,
            fill=None,
            hoverinfo="skip",
        )
    )

    # fit filled stop
    fig.add_trace(
        go.Scatter(
            x=fit["z"],
            y=fit["mu"] - fit["std"],
            mode="lines",
            line_width=0,
            fill="tonexty",
            fillcolor=green_color_transparent,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    return fig


#%%


def update_raw_count_plots(results, click_data, forward_reverse):
    if click_data is not None:

        shortname, tax_id = dashboard.utils.get_shortname_tax_id_from_click_data(
            results,
            click_data,
        )

        group = results.get_single_count_group(
            shortname,
            tax_id,
            forward_reverse,
        )
        fit = results.get_single_fit_prediction(
            shortname,
            tax_id,
            forward_reverse,
        )
        fig = dashboard.figures.plot_group(results, group, fit, forward_reverse)
        return fig
    else:
        raise PreventUpdate
