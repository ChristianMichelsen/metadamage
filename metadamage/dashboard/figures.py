# Scientific Library
import numpy as np
import pandas as pd

# Third Party
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# First Party
from metadamage import counts, dashboard


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


def plot_fit_results(fit_results, df_fit_results=None):

    if df_fit_results is None:
        df_fit_results = fit_results.df_fit_results

    fig = px.scatter(
        df_fit_results,
        x="LR",
        y="D_max",
        size="size",
        color="shortname",
        hover_name="shortname",
        # size_max=marker_size_max,
        # opacity=1,
        color_discrete_map=fit_results.d_cmap,
        custom_data=fit_results.custom_data_columns,
        range_x=fit_results.ranges["LR"],
        range_y=[0, 1],
        render_mode="webgl",
        symbol="shortname",
        symbol_map=fit_results.d_symbols,
    )

    fig.update_traces(
        hovertemplate=fit_results.hovertemplate,
        marker_line_width=0,
        marker_sizeref=2.0
        * fit_results.max_of_size
        / (fit_results.marker_size_max ** 2),
    )

    fig.update_layout(
        xaxis_title=r"$\Large \lambda_\mathrm{LR}$",
        yaxis_title=r"$\Large D_\mathrm{max}$",
        # legend_title="Files",
        showlegend=False,
    )

    fig.for_each_trace(
        lambda trace: set_opacity_for_trace(
            trace,
            method="sqrt",
            scale=20 / df_fit_results.shortname.nunique(),
            opacity_min=0.001,
            opacity_max=0.8,
        )
    )

    return fig


#%%


def plotly_histogram(
    fit_results,
    data,
    filename,
    column,
    bins=50,
    density=True,
    range=None,
    showlegend=True,
):
    data = data[~np.isnan(data)]
    binned = np.histogram(data, bins=bins, density=density, range=range)
    trace = go.Scatter(
        x=binned[1],
        y=binned[0],
        mode="lines",
        name=filename,
        legendgroup=filename,
        line_shape="hv",  # similar to 'mid' in matplotlib,
        showlegend=showlegend,
        marker_color=fit_results.d_cmap[filename],
        hovertemplate=(
            "<b>" + f"{filename}" + "</b><br><br>"
            f"{column}" + ": %{x:5.2f}<br>"
            "Counts: %{y}<br>"
            "<extra></extra>"
        ),
    )

    return trace, np.max(binned[0])


def plot_histograms(fit_results, df_fit_results=None):

    if df_fit_results is None:
        df_fit_results = fit_results.df_fit_results

    fig = make_subplots(rows=2, cols=4)

    showlegend = True

    it = fit_results.iterate_over_scatter_columns(N_cols=4)
    for column, row, col in it:
        highest_y_max = 0
        for filename, group in df_fit_results.groupby("shortname", sort=False):
            trace, y_max = plotly_histogram(
                fit_results=fit_results,
                data=group[column],
                filename=filename,
                column=column,
                bins=50,
                density=False,
                range=fit_results.ranges[column],
                showlegend=showlegend,
            )
            fig.add_trace(trace, col=col, row=row)
            if y_max > highest_y_max:
                highest_y_max = y_max

        showlegend = False
        fig.update_xaxes(title_text=fit_results.labels[column], row=row, col=col)
        fig.update_yaxes(rangemode="nonnegative")
        if col == 1:
            fig.update_yaxes(title_text="Counts", row=row, col=col)

    fig.update_layout(
        font_size=12,
        legend_title="Files",
    )

    return fig


#%%


def set_opacity_for_trace_scatter_matrix(trace):
    N = len(trace.dimensions[0].values)
    # opacity = 1 / np.sqrt(N)
    opacity = 1 / N ** 0.75
    trace.update(marker_opacity=opacity)


def plot_scatter_matrix(fit_results, df_fit_results=None):

    if df_fit_results is None:
        df_fit_results = fit_results.df_fit_results

    fig = px.scatter_matrix(
        df_fit_results,
        dimensions=fit_results.columns_scatter,
        color="shortname",
        hover_name="shortname",
        size_max=10,
        color_discrete_map=fit_results.d_cmap,
        labels=fit_results.labels,
        custom_data=fit_results.custom_data_columns,
        symbol="shortname",
        symbol_map=fit_results.d_symbols,
    )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        hovertemplate=fit_results.hovertemplate,
    )

    fig.for_each_trace(set_opacity_for_trace_scatter_matrix)

    # manually set ranges for scatter matrix
    iterator = enumerate(fit_results.columns_scatter)
    ranges = {i + 1: fit_results.ranges[col] for i, col in iterator}
    for axis in ["xaxis", "yaxis"]:
        fig.update_layout({axis + str(k): {"range": v} for k, v in ranges.items()})

    fig.update_layout(
        font_size=12,
        legend_title="Files",
    )

    return fig


#%%


def plot_forward_reverse(fit_results, df_fit_results=None):

    if df_fit_results is None:
        df_fit_results = fit_results.df_fit_results

    N_rows = 3
    N_cols = 2
    subtitles = list(fit_results.columns_scatter_forward_reverse.values())
    fig = make_subplots(rows=N_rows, cols=N_cols, subplot_titles=subtitles)

    for it in fit_results.iterate_over_scatter_columns_forward_reverse(N_cols):
        dimension, row, column, showlegend, forward, reverse = it

        for filename, group in df_fit_results.groupby("shortname", sort=False):

            if len(group) == 0:
                continue

            kwargs = dict(
                name=filename,
                mode="markers",
                legendgroup=filename,
                marker=dict(color=fit_results.d_cmap[filename]),  # opacity=0.2
                hovertemplate=fit_results.hovertemplate,
                customdata=np.array(group[fit_results.custom_data_columns].values),
            )

            fig.add_trace(
                go.Scattergl(
                    x=group[forward],
                    y=group[reverse],
                    showlegend=showlegend,
                    **kwargs,
                ),
                row=row,
                col=column,
            )

        fig.update_xaxes(range=fit_results.ranges[forward], row=row, col=column)
        fig.update_yaxes(range=fit_results.ranges[reverse], row=row, col=column)

    # Update xaxis properties
    fig.update_xaxes(title_text="Forward")
    fig.update_yaxes(title_text="Reverse")

    fig.for_each_trace(lambda trace: set_opacity_for_trace(trace, method="sqrt"))

    fig.update_layout(font_size=12, legend_title="Files")

    return fig


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


def _plot_specific_mismatch(
    fig,
    df,
    x,
    mismatch,
    i,
    highlight_mismatch,
    show_legend,
    cmap,
    row,
):

    ref = mismatch[0]
    obs = mismatch[-1]

    y = df[f"{ref}{obs}"].values
    reference_columns = counts.get_reference_columns(df, ref)
    M = df[reference_columns].sum(axis=1).values

    customdata = np.stack([y, M]).T

    f = y / M

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=x,
            y=f,
            mode="markers",
            name=mismatch,
            legendgroup=mismatch,
            showlegend=show_legend,
            marker_color=cmap[i],
            marker_size=7 if mismatch in highlight_mismatch else 5,
            marker_opacity=1 if mismatch in highlight_mismatch else 0.75,
            customdata=customdata,
            hovertemplate=(
                "<b>" + f"{mismatch}" + "</b><br><br>"
                f"{mismatch}" + ": %{customdata[0]:.3s}<br>"
                f"{ref}" + ":   %{customdata[1]:.3s}<br>"
                "Fraction: %{y:.3f}<br>"
                "Position: %{x}<br>"
                "<extra></extra>"
            ),
        ),
        col=1,
        row=row,
    )


def _plot_mismatch_fit_error_bars(
    fig,
    x,
    y,
    row,
    customdata_errors,
    show_legend,
    green_color,
    hovertemplate,
):

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type="data",
                symmetric=False,
                array=customdata_errors[:, 0],
                arrayminus=customdata_errors[:, 1],
                thickness=2,
                width=5,
            ),
            marker_symbol="line-ns",
            mode="markers",
            name="Fit",
            legendgroup="Fit",
            showlegend=show_legend,
            marker_color=green_color,
            customdata=customdata_errors,
            hovertemplate=hovertemplate,
        ),
        col=1,
        row=row,
    )


def _plot_mismatch_fit_filled(
    fig,
    x,
    y,
    errors,
    row,
    green_color,
    green_color_transparent,
    customdata_errors,
    hovertemplate,
    show_legend,
):

    # fix when filling between lines when the lines include nans
    not_mask = ~np.isnan(y)

    fig.add_trace(
        go.Scatter(
            name="Fit",
            x=x[not_mask],
            y=y[not_mask],
            mode="lines",
            line_color=green_color,
            customdata=customdata_errors,
            hovertemplate=hovertemplate,
            showlegend=show_legend,
        ),
        col=1,
        row=row,
    )

    fig.add_trace(
        go.Scatter(
            x=x[not_mask],
            y=errors[1, not_mask],
            mode="lines",
            line_width=0,
            showlegend=False,
            hoverinfo="skip",
        ),
        col=1,
        row=row,
    )

    fig.add_trace(
        go.Scatter(
            x=x[not_mask],
            y=errors[0, not_mask],
            line_width=0,
            mode="lines",
            fillcolor=green_color_transparent,
            fill="tonexty",
            showlegend=False,
            hoverinfo="skip",
        ),
        col=1,
        row=row,
    )


def _plot_mismatch_fit(
    fig,
    fit,
    x,
    row,
    show_legend,
    is_forward,
    use_error_bars,
):

    customdata_errors = np.stack(
        [
            fit["hdpi_upper"] - fit["median"],
            fit["median"] - fit["hdpi_lower"],
        ]
    ).T

    if is_forward:
        y = fit["median"][:15].values
        customdata_errors = customdata_errors[:15, :]
        errors = fit[["hdpi_lower", "hdpi_upper"]].values.T[:, :15]
    else:
        y = fit["median"][15:].values
        customdata_errors = customdata_errors[15:, :]
        errors = fit[["hdpi_lower", "hdpi_upper"]].values.T[:, 15:]

    green_color = "#2CA02C"
    green_color_transparent = dashboard.utils.hex_to_rgb(green_color, opacity=0.2)

    hovertemplate = (
        "<b>Fit</b><br><br>"
        "Fraction: %{y:.3f}<br>"
        "Error (68% HDPI) High: +%{customdata[0]:.3f}<br>"
        "Error (68% HDPI) Low:  -%{customdata[1]:.3f}<br>"
        "Position: %{x}<br>"
        "<extra></extra>"
    )

    if use_error_bars:
        _plot_mismatch_fit_error_bars(
            fig,
            x,
            y,
            row,
            customdata_errors,
            show_legend,
            green_color,
            hovertemplate,
        )

    else:
        _plot_mismatch_fit_filled(
            fig,
            x,
            y,
            errors,
            row,
            green_color,
            green_color_transparent,
            customdata_errors,
            hovertemplate,
            show_legend,
        )


def _plot_count_fraction(
    fig,
    df,
    chosen_mismatch_columns,
    fit=None,
    is_forward=True,
    use_error_bars=False,
):

    cmap = [
        "#1F77B4",  # blue
        "#D62728",  # red
        # "#2CA02C",  # green
        "#FF7F0E",  # orange
        "#9467BD",  # purple
        "#8C564B",  # brown
        "#E377C2",  # pink
        "#7F7F7F",  # grey
        "#BCBD22",  # camo
        "#17BECF",  # turquoise
    ]

    x = df.position.values
    if is_forward:
        row = 1
        show_legend = True
        highlight_mismatch = ["C→T"]
    else:
        x = -x
        row = 2
        show_legend = False
        highlight_mismatch = ["G→A"]

    for i, mismatch in enumerate(chosen_mismatch_columns):
        _plot_specific_mismatch(
            fig,
            df,
            x,
            mismatch,
            i,
            highlight_mismatch,
            show_legend,
            cmap,
            row,
        )

    #%%

    if fit is not None:
        _plot_mismatch_fit(
            fig,
            fit,
            x,
            row,
            show_legend,
            is_forward,
            use_error_bars,
        )

    #%%

    if not is_forward:
        fig.update_xaxes(title_text=r"$\text{Position}, z$", row=row, col=1)

    # x_ticks = np.arange(1, 15+1, 1)
    # if is_forward:
    #     x_ticks_names = [tick if tick%2==1 else '' for tick in x_ticks ]
    # else:
    #     x_ticks_names = [-tick if tick%2==1 else '' for tick in x_ticks ]

    x_ticks = np.arange(1, 15 + 2, 2)
    x_ticks_names = x_ticks if is_forward else -x_ticks

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_ticks,
        ticktext=x_ticks_names,
        row=row,
        col=1,
    )

    fig.update_yaxes(
        title="Fraction",
        rangemode="nonnegative",
    )


def plot_count_fraction(group, chosen_mismatch_columns=None, fit=None):

    if chosen_mismatch_columns is None:
        chosen_mismatch_columns = ["C→T", "G→A"]

    group_forward = group.query(f"position >= 0")
    group_reverse = group.query(f"position <= 0")

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Forward", "Reverse"])

    _plot_count_fraction(
        fig,
        df=group_forward,
        chosen_mismatch_columns=chosen_mismatch_columns,
        fit=fit,
        is_forward=True,
        # use_error_bars=False,
    )

    _plot_count_fraction(
        fig,
        df=group_reverse,
        chosen_mismatch_columns=chosen_mismatch_columns,
        fit=fit,
        is_forward=False,
        # use_error_bars=False,
    )

    fig.update_layout(
        # title_text="Mismatch Fractions",
        legend_title="Mismatches",
    )

    return fig
