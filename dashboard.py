import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.express as px
import dashboard_helper
import pandas as pd
from pathlib import Path
import numpy as np

from dash.exceptions import PreventUpdate


external_stylesheets = [dbc.themes.BOOTSTRAP]
external_scripts = [
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
    "MathJax.js?config=TeX-MML-AM_CHTML",
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    external_scripts=external_scripts,
    title="metaDashboard",
    update_title="Updating...",
)

# to allow custom css
app.scripts.config.serve_locally = True

# First Party
from metadamage import dashboard

dashboard.utils.set_custom_theme()
# reload(dashboard)


#%%

graph_kwargs = dict(
    config={
        "displaylogo": False,
        "doubleClick": "reset",
        "showTips": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
    },
    # # https://css-tricks.com/fun-viewport-units/
    # style={"width": "100%", "height": "55vh"},
)


graph_kwargs_no_buttons = dict(
    config={
        "displaylogo": False,
        "doubleClick": "reset",
        "showTips": True,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
            "toImage",
        ],
    },
)


#%%


fit_results = dashboard.fit_results.FitResults(
    folder=Path("./data/out/"),
    use_memoization=False,
)

# x = x

# fit_results.set_marker_size(marker_transformation="log10", marker_size_max=8)
# df = fit_results.df_fit_results

# df = pd.read_csv("https://plotly.github.io/datasets/country_indicators.csv")
columns = list(fit_results.df_fit_results.columns)
exclude_cols_including = [
    "tax",
    "valid",
    "LR_P",
    "LR_n_sigma",
    "size",
    "shortname",
    "_log10",
    "_sqrt",
]
columns = [
    column
    for column in columns
    if not any([word in column for word in exclude_cols_including])
]
exclude_cols = ["_LR", "_forward_LR", "_reverse_LR"]
columns = [column for column in columns if column not in exclude_cols]


d_columns_latex = {
    "LR": r"$\lambda_\text{LR}$",
    "D_max": r"$D_\text{max}$",
    "D_max_std": r"$\sigma_{D_\text{max}}$",
    "q": r"$q$",
    "q_std": r"$\sigma_q$",
    "phi": r"$\phi$",
    "phi_std": r"$\sigma_\phi$",
    "A": r"$A$",
    "A_std": r"$\sigma_A$",
    "c": r"$c$",
    "c_std": r"$\sigma_c$",
    "rho_Ac": r"$\rho_{A, c}$",
    "LR_P": r"$\text{P}_\lambda$",
    "LR_n_sigma": r"$\sigma_\lambda$",
    "asymmetry": r"$\text{asymmetry}$",
    #
    "forward_LR": r"$ \lambda_\text{LR} \,\, \text{(forward)}$",
    "forward_D_max": r"$ D\text{max} \,\, \text{(forward)}$",
    "forward_D_max_std": r"$ \sigma_{D_\text{max}} \,\, \text{(forward)}$",
    "forward_q": r"$ q \,\, \text{(forward)}$",
    "forward_q_std": r"$ \sigma_q \,\, \text{(forward)}$",
    "forward_phi": r"$ \phi \,\, \text{(forward)}$",
    "forward_phi_std": r"$ \sigma_\phi \,\, \text{(forward)}$",
    "forward_A": r"$ A \,\, \text{(forward)}$",
    "forward_A_std": r"$ \sigma_A \,\, \text{(forward)}$",
    "forward_c": r"$ c \,\, \text{(forward)}$",
    "forward_c_std": r"$ \sigma_c \,\, \text{(forward)}$",
    "forward_rho_Ac": r"$ \rho_{A, c} \,\, \text{(forward)}$",
    "forward_LR_P": r"$ \text{P}_\lambda \,\, \text{(forward)}$",
    "forward_LR_n_sigma": r"$ \sigma_\lambda \,\, \text{(forward)}$",
    #
    "reverse_LR": r"$ \lambda_\text{LR} \,\, \text{(reverse)}$",
    "reverse_D_max": r"$ D\text{max} \,\, \text{(reverse)}$",
    "reverse_D_max_std": r"$ \sigma_{D_\text{max}} \,\, \text{(reverse)}$",
    "reverse_q": r"$ q \,\, \text{(reverse)}$",
    "reverse_q_std": r"$ \sigma_q \,\, \text{(reverse)}$",
    "reverse_phi": r"$ \phi \,\, \text{(reverse)}$",
    "reverse_phi_std": r"$ \sigma_\phi \,\, \text{(reverse)}$",
    "reverse_A": r"$ A \,\, \text{(reverse)}$",
    "reverse_A_std": r"$ \sigma_A \,\, \text{(reverse)}$",
    "reverse_c": r"$ c \,\, \text{(reverse)}$",
    "reverse_c_std": r"$ \sigma_c \,\, \text{(reverse)}$",
    "reverse_rho_Ac": r"$ \rho_{A, c} \,\, \text{(reverse)}$",
    "reverse_LR_P": r"$ \text{P}_\lambda \,\, \text{(reverse)}$",
    "reverse_LR_n_sigma": r"$ \sigma_\lambda \,\, \text{(reverse)}$",
    #
    "N_alignments": r"$N_\text{alignments}$",
    #
    "N_z1_forward": r"$N_{z=1} \,\, \text{(forward)}$",
    "N_z1_reverse": r"$N_{z=1} \,\, \text{(reverse)}$",
    #
    "N_sum_total": r"$\sum_i N_i$",
    "N_sum_forward": r"$\sum_i N_i \,\, \text{(forward)}$",
    "N_sum_reverse": r"$\sum_i N_i \,\, \text{(reverse)}$",
    #
    "N_min": r"$\text{min} N_i$",
    #
    "k_sum_total": r"$\sum_i k_i$",
    "k_sum_forward": r"$\sum_i k_i \,\, \text{(forward)}$",
    "k_sum_reverse": r"$\sum_i k_i \,\, \text{(reverse)}$",
    #
    "Bayesian_D_max": r"$D_\text{max} \,\, \text{(Bayesian)}$",
    "Bayesian_D_max_std": r"$\sigma_{D_\text{max}} \,\, \text{(Bayesian)}$",
    "Bayesian_n_sigma": r"$n_\sigma \,\, \text{(Bayesian)}$",
    "Bayesian_A": r"$A \,\, \text{(Bayesian)}$",
    "Bayesian_q": r"$q \,\, \text{(Bayesian)}$",
    "Bayesian_c": r"$c \,\, \text{(Bayesian)}$",
    "Bayesian_phi": r"$\phi \,\, \text{(Bayesian)}$",
    #
    "D_max_significance": r"$Z_{D_\text{max}}$",
    "rho_Ac_abs": r"$|\rho_{A, c}|$",
}

# x = x

# (1) No sidebars, (2) Only left filter sidebar,
# (3) Only right plot sidebar, (4) Both sidebars
start_configuration_id = 1

sidebar_filter_width = 20  # in %
sidebar_plot_width = 35  # in %
content_main_margin = 1  # in %


configurations = dashboard_helper.get_configurations(
    sidebar_filter_width,
    sidebar_plot_width,
    content_main_margin,
)
start_configuration = configurations[start_configuration_id]


navbar = dbc.NavbarSimple(
    [
        dbc.Button(
            "Toggle Filter",
            outline=True,
            color="secondary",
            className="mr-1",
            id="btn_toggle_filter",
        ),
        dbc.Button(
            "Toggle Plot",
            outline=True,
            color="secondary",
            className="mr-1",
            id="btn_toggle_plot",
        ),
    ],
    brand="metaDashboard",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)


dropdown_x_axis = dcc.Dropdown(
    id="xaxis_column",
    options=[{"label": i, "value": i} for i in columns],
    value="LR",
)

lin_log_scale_x_axis = dcc.RadioItems(
    id="xaxis_type",
    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
    value="Linear",
    labelStyle={"display": "inline-block"},
)

dropdown_y_axis = dcc.Dropdown(
    id="yaxis_column",
    options=[{"label": i, "value": i} for i in columns],
    value="D_max",
)

lin_log_scale_y_axis = dcc.RadioItems(
    id="yaxis_type",
    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
    value="Linear",
    labelStyle={"display": "inline-block"},
)

div_x_axis = html.Div(
    [dropdown_x_axis, lin_log_scale_x_axis],
    style={"width": "48%", "display": "inline-block"},
)

div_y_axis = html.Div(
    [dropdown_y_axis, lin_log_scale_y_axis],
    style={
        "width": "48%",
        "float": "right",
        "display": "inline-block",
    },
)


content_main = html.Div(
    html.Div(
        [
            html.Div([div_x_axis, div_y_axis]),
            dcc.Graph(id="indicator_graphic", **graph_kwargs),
            # slider,
        ]
    ),
    id="content_main",
    style=start_configuration.style_content_main,
)


#%%


filter_dropdown_file = dbc.FormGroup(
    [
        html.Br(),
        dbc.Col(html.H3("Input samples"), width=12),
        dashboard.elements.get_dropdown_file_selection(
            fit_results=fit_results,
            id="sidebar_filter_dropdown_shortnames",
            shortnames_to_show="each",  # one for each first letter in shortname
        ),
    ]
)

filters_collapse_files = html.Div(
    [
        dbc.Button(
            "Filter Files",
            id="filters_toggle_files_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            filter_dropdown_file,
            id="filters_dropdown_files",
            is_open=False,
        ),
    ]
)


#%%


# Standard Library
import itertools

filter_tax_id = dbc.Row(
    [
        dbc.Col(html.Br(), width=12),
        dbc.Col(html.H3("Specific taxa"), width=12),
        dbc.Col(
            dbc.FormGroup(
                [
                    dcc.Dropdown(
                        id="tax_id_filter_input",
                        options=[
                            {"label": tax, "value": tax}
                            for tax in itertools.chain.from_iterable(
                                [
                                    fit_results.all_tax_ranks,
                                    fit_results.all_tax_names,
                                    fit_results.all_tax_ids,
                                ]
                            )
                        ],
                        clearable=True,
                        multi=True,
                        placeholder="Select taxas...",
                    ),
                ],
            ),
            width=12,
        ),
        dbc.Col(html.Br(), width=12),
        dbc.Col(html.H3("Taxanomic descendants"), width=12),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Input(
                        id="tax_id_filter_input_descendants",
                        placeholder="Input goes here...",
                        type="text",
                        autoComplete="off",
                    ),
                ]
            ),
            width=12,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Checklist(
                        options=[
                            {"label": "Include subspecies", "value": True},
                        ],
                        value=[True],
                        id="tax_id_filter_subspecies",
                    ),
                ]
            ),
            width=12,
        ),
        dbc.Col(html.P(id="tax_id_filter_counts_output"), width=12),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Button(
                        "Plot", id="tax_id_plot_button", color="primary", block=True
                    ),
                ]
            ),
        ),
    ],
    justify="between",
    form=True,
)


filters_collapse_tax_id = html.Div(
    [
        dbc.Button(
            "Filter Tax IDs",
            id="filters_toggle_tax_ids_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            filter_tax_id,
            id="filters_dropdown_tax_ids",
            is_open=False,
        ),
    ]
)

#%%


slider_names = [
    "LR",
    "D_max",
    "q",
    "phi",
    "N_alignments",
    "k_sum_total",
    "N_sum_total",
]

filters_collapse_ranges = html.Div(
    [
        dbc.Button(
            "Filter Fit Results",
            id="filters_toggle_ranges_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            [
                html.Br(),
                dbc.Col(
                    html.H3("Fit results"),
                    width=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown_slider",
                        options=[
                            {"label": shortname, "value": shortname}
                            for shortname in slider_names
                        ],
                        value=[],
                        multi=True,
                        placeholder="Select a variable to filter on...",
                        optionHeight=30,
                    ),
                    width=12,
                ),
                dbc.Col(
                    id="dynamic_slider_container",
                    children=[],
                    width=12,
                ),
            ],
            id="filters_dropdown_ranges_button",
            is_open=False,
        ),
    ]
)


#%%

sidebar_filter = html.Div(
    [
        html.H2("Filter", className="display-4"),
        html.Hr(),
        html.P("filter here", className="lead"),
        dbc.Form(
            [
                filters_collapse_files,
                html.Hr(),
                filters_collapse_tax_id,
                html.Hr(),
                filters_collapse_ranges,
            ]
        ),
    ],
    id="sidebar_filter",
    style=start_configuration.style_sidebar_filter,
)

#%%


sidebar_plot_combined_graph = dbc.FormGroup(
    [
        dcc.Graph(
            figure=dashboard.figures.create_empty_figure(),
            id="graph_plot_data",
            **graph_kwargs_no_buttons,
        ),
    ]
)

sidebar_plot_combined = html.Div(
    [
        dbc.Button(
            "Combined",
            id="sidebar_plot_toggle_combined",
            color="secondary",
            block=True,
            outline=False,
            size="lg",
        ),
        dbc.Collapse(
            sidebar_plot_combined_graph,
            id="sidebar_plot_combined",
            is_open=True,
        ),
    ]
)


@app.callback(
    Output("sidebar_plot_combined", "is_open"),
    Output("sidebar_plot_toggle_combined", "outline"),
    Input("sidebar_plot_toggle_combined", "n_clicks"),
    State("sidebar_plot_combined", "is_open"),
)
def toggle_collapse_plot_combined(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, False


sidebar_plot_forward_reverse_graph = dbc.FormGroup(
    [
        dcc.Graph(
            figure=dashboard.figures.create_empty_figure(),
            id="graph_plot_data_forward",
            style={"height": "20vh"},
            **graph_kwargs_no_buttons,
        ),
        dcc.Graph(
            figure=dashboard.figures.create_empty_figure(),
            id="graph_plot_data_reverse",
            style={"height": "20vh"},
            **graph_kwargs_no_buttons,
        ),
    ]
)

sidebar_plot_forward_reverse = html.Div(
    [
        dbc.Button(
            "Forward / Reverse",
            id="sidebar_plot_toggle_forward_reverse",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            sidebar_plot_forward_reverse_graph,
            id="sidebar_plot_forward_reverse",
            is_open=False,
        ),
    ]
)


@app.callback(
    Output("sidebar_plot_forward_reverse", "is_open"),
    Output("sidebar_plot_toggle_forward_reverse", "outline"),
    Input("sidebar_plot_toggle_forward_reverse", "n_clicks"),
    State("sidebar_plot_forward_reverse", "is_open"),
)
def toggle_collapse_plot_forward_reverse(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


sidebar_plot = html.Div(
    [
        html.H2("Plot", className="display-4"),
        html.Hr(),
        sidebar_plot_combined,
        html.Hr(),
        sidebar_plot_forward_reverse,
    ],
    id="sidebar_plot",
    style=start_configuration.style_sidebar_plot,
)

app.layout = html.Div(
    [
        # dcc.Store(id="store"),
        dcc.Store(id="sidebar_plot_state"),
        dcc.Store(id="sidebar_filter_state"),
        navbar,
        sidebar_filter,
        content_main,
        sidebar_plot,
        dbc.Modal(
            [
                dbc.ModalHeader("Filtering Error"),
                dbc.ModalBody(
                    "Too restrictive filtering, no points left to plot. "
                    "Please choose a less restrictive filtering."
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="modal_close_button", className="ml-auto")
                ),
            ],
            centered=True,
            id="modal",
        ),
    ],
)

#%%


def get_shortname_tax_id_from_click_data(fit_results, click_data):
    try:
        shortname = fit_results.parse_click_data(click_data, column="shortname")
        tax_id = fit_results.parse_click_data(click_data, column="tax_id")
    except KeyError:
        raise PreventUpdate
    return shortname, tax_id


import plotly.graph_objects as go


def plot_group(group, fit=None, forward_reverse=""):

    custom_data_columns = [
        "direction",
        # "z",
        "k",
        "N",
        # "shortname",
        # "tax_name",
        # "tax_rank",
        # "tax_id",
    ]

    # hovertemplate = (
    #     "<b>Direction: %{customdata[0]}</b><br><br>"
    #     "<b>Data</b>: <br>"
    #     "    z:    %{customdata[1]:4d} <br>"
    #     "    k:    %{customdata[2]:2.3s} <br>"
    #     "    N:    %{customdata[3]:2.3s} <br><br>"
    #     # "<b>Sample</b>: <br>"
    #     # "    Name: %{customdata[4]} <br><br>"
    #     # "<b>Tax</b>: <br>"
    #     # "    Name: %{customdata[5]} <br>"
    #     # "    Rank: %{customdata[6]} <br>"
    #     # "    ID:   %{customdata[7]} <br><br>"
    #     "<extra></extra>"
    # )

    hovertemplate = (
        "<b>Direction: %{customdata[0]}</b><br>"
        # "z: %{customdata[1]:8d} <br>"
        "k: %{customdata[1]:8d} <br>"
        "N: %{customdata[2]:8d} <br>"
        "<extra></extra>"
    )

    fig = px.scatter(
        group,
        x="z",
        y="f",
        color="direction",
        color_discrete_map=fit_results.d_cmap_fit,
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

    green_color = fit_results.d_cmap_fit["Fit"]
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
            hovertemplate=fit_results.hovertemplate_fit,
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


# shortname = "KapK-12-1-24-Ext-1-Lib-1-Index2"
# tax_id = 20802


@app.callback(
    Output("graph_plot_data", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_dropdowns_based_on_click_data(click_data):
    if click_data is not None:

        shortname, tax_id = get_shortname_tax_id_from_click_data(
            fit_results, click_data
        )
        forward_reverse = ""
        group = fit_results.get_single_count_group(shortname, tax_id, forward_reverse)
        fit = fit_results.get_single_fit_prediction(shortname, tax_id, forward_reverse)
        fig = plot_group(group, fit, forward_reverse)
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output("graph_plot_data_forward", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_dropdowns_based_on_click_data(click_data):
    if click_data is not None:

        shortname, tax_id = get_shortname_tax_id_from_click_data(
            fit_results, click_data
        )
        forward_reverse = "Forward"
        group = fit_results.get_single_count_group(shortname, tax_id, forward_reverse)
        fit = fit_results.get_single_fit_prediction(shortname, tax_id, forward_reverse)
        fig = plot_group(group, fit, forward_reverse)
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output("graph_plot_data_reverse", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_dropdowns_based_on_click_data(click_data):
    if click_data is not None:

        shortname, tax_id = get_shortname_tax_id_from_click_data(
            fit_results, click_data
        )
        forward_reverse = "Reverse"
        group = fit_results.get_single_count_group(shortname, tax_id, forward_reverse)
        fit = fit_results.get_single_fit_prediction(shortname, tax_id, forward_reverse)
        fig = plot_group(group, fit, forward_reverse)
        return fig
    else:
        raise PreventUpdate


#%%


def include_subspecies(subspecies):
    if len(subspecies) == 1:
        return True
    return False


def append_to_list_if_exists(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


def apply_tax_id_filter(d_filter, tax_id_filter_input):
    if tax_id_filter_input is None or len(tax_id_filter_input) == 0:
        return None

    for tax in tax_id_filter_input:
        if tax in fit_results.all_tax_ids:
            append_to_list_if_exists(d_filter, "tax_ids", tax)
        elif tax in fit_results.all_tax_names:
            append_to_list_if_exists(d_filter, "tax_names", tax)
        elif tax in fit_results.all_tax_ranks:
            append_to_list_if_exists(d_filter, "tax_ranks", tax)
        else:
            raise AssertionError(f"Tax {tax} could not be found. ")


from metadamage import taxonomy


def apply_tax_id_descendants_filter(d_filter, tax_name, tax_id_filter_subspecies):
    if tax_name is None:
        return None

    tax_ids = taxonomy.extract_descendant_tax_ids(
        tax_name,
        include_subspecies=include_subspecies(tax_id_filter_subspecies),
    )
    N_tax_ids = len(tax_ids)
    if N_tax_ids != 0:
        if "tax_id" in d_filter:
            d_filter["tax_ids"].extend(tax_ids)
        else:
            d_filter["tax_ids"] = tax_ids


def make_figure(
    df,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
):

    fig = px.scatter(
        df,
        x=xaxis_column_name,
        y=yaxis_column_name,
        size="size",
        color="shortname",
        hover_name="shortname",
        color_discrete_map=fit_results.d_cmap,
        custom_data=fit_results.custom_data_columns,
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
        xaxis_title=xaxis_column_name,
        yaxis_title=yaxis_column_name,
        showlegend=False,
        # legend_title="Files",
    )

    fig.for_each_trace(
        lambda trace: dashboard.figures.set_opacity_for_trace(
            trace,
            method="sqrt",
            scale=20 / df.shortname.nunique(),
            opacity_min=0.3,
            opacity_max=0.95,
        )
    )

    fig.update_xaxes(
        title=d_columns_latex[xaxis_column_name],
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=d_columns_latex[yaxis_column_name],
        type="linear" if yaxis_type == "Linear" else "log",
    )

    return fig


@app.callback(
    Output("indicator_graphic", "figure"),
    Output("modal", "is_open"),
    Input("sidebar_filter_dropdown_shortnames", "value"),
    Input("tax_id_filter_input", "value"),
    Input("tax_id_plot_button", "n_clicks"),
    Input({"type": "dynamic_slider", "index": ALL}, "value"),
    Input("xaxis_column", "value"),
    Input("yaxis_column", "value"),
    Input("xaxis_type", "value"),
    Input("yaxis_type", "value"),
    Input("modal_close_button", "n_clicks"),
    State({"type": "dynamic_slider", "index": ALL}, "id"),
    State("tax_id_filter_input_descendants", "value"),
    State("tax_id_filter_subspecies", "value"),
    State("modal", "is_open"),
)
def update_graph(
    dropdown_file_selection,
    tax_id_filter_input,
    tax_id_button,
    slider_values,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    n_clicks_modal,
    slider_ids,
    tax_id_filter_input_descendants,
    tax_id_filter_subspecies,
    modal_is_open,
):

    # if modal is open and the "close" button is clicked, close down modal
    if n_clicks_modal and modal_is_open:
        return dash.no_update, False

    # if no files selected
    if not dropdown_file_selection:
        raise PreventUpdate

    # fit_results.set_marker_size(marker_transformation, marker_size_max)

    d_filter = {"shortnames": dropdown_file_selection}

    slider_names = [id["index"] for id in slider_ids]
    for shortname, values in zip(slider_names, slider_values):
        d_filter[shortname] = values

    apply_tax_id_filter(
        d_filter,
        tax_id_filter_input,
    )

    apply_tax_id_descendants_filter(
        d_filter,
        tax_id_filter_input_descendants,
        tax_id_filter_subspecies,
    )

    df_fit_results_filtered = fit_results.filter(d_filter)

    # raise modal warning if no results due to too restrictive filtering
    if len(df_fit_results_filtered) == 0:
        return dash.no_update, True

    fig = make_figure(
        df=df_fit_results_filtered,
        xaxis_column_name=xaxis_column_name,
        yaxis_column_name=yaxis_column_name,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
    )

    return fig, dash.no_update


#%%


def key_is_in_list_case_insensitive(lst, key):
    return any([key.lower() in s.lower() for s in lst])


@app.callback(
    Output("sidebar_filter_dropdown_shortnames", "value"),
    Input("sidebar_filter_dropdown_shortnames", "value"),
)
def update_dropdown_when_Select_All(dropdown_file_selection):
    if key_is_in_list_case_insensitive(dropdown_file_selection, "Select all"):
        dropdown_file_selection = fit_results.shortnames
    elif key_is_in_list_case_insensitive(dropdown_file_selection, "Default selection"):
        dropdown_file_selection = dashboard.elements.get_shortnames_each(
            fit_results.shortnames
        )

    dropdown_file_selection = list(sorted(dropdown_file_selection))

    return dropdown_file_selection


#%%


def get_id_dict(child):
    return child["props"]["id"]


def find_index_in_children(children, id_type, search_index):
    for i, child in enumerate(children):
        d_id = get_id_dict(child)
        if d_id["type"] == id_type and d_id["index"] == search_index:
            return i


def get_current_names(current_ids):
    return [x["index"] for x in current_ids if x]


def slider_is_added(current_names, dropdown_names):
    "Returns True if a new slider is added, False otherwise"
    return set(current_names).issubset(dropdown_names)


def get_name_of_added_slider(current_names, dropdown_names):
    return list(set(dropdown_names).difference(current_names))[0]


def get_name_of_removed_slider(current_names, dropdown_names):
    return list(set(current_names).difference(dropdown_names))[0]


def remove_name_from_children(column, children, id_type):
    " Given a column, remove the corresponding child element from children"
    index = find_index_in_children(children, id_type=id_type, search_index=column)
    children.pop(index)


from metadamage import utils


def get_slider_name(column, low_high):
    if isinstance(low_high, dict):
        low = low_high["min"]
        high = low_high["max"]
    elif isinstance(low_high, (tuple, list)):
        low = low_high[0]
        high = low_high[1]

    if column in dashboard.utils.log_transform_columns:
        low = dashboard.utils.log_transform_slider(low)
        high = dashboard.utils.log_transform_slider(high)

    low = utils.human_format(low)
    high = utils.human_format(high)

    return f"{column}: [{low}, {high}]"


def make_new_slider(column, id_type, N_steps=100):

    d_range_slider = dashboard.elements.get_range_slider_keywords(
        fit_results,
        column=column,
        N_steps=N_steps,
    )

    return dbc.Container(
        [
            dbc.Row(html.Br()),
            dbc.Row(
                html.P(
                    get_slider_name(column, d_range_slider),
                    id={"type": "dynamic_slider_name", "index": column},
                ),
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.RangeSlider(
                        id={"type": "dynamic_slider", "index": column},
                        **d_range_slider,
                    ),
                    width=12,
                ),
            ),
        ],
        id={"type": id_type, "index": column},
    )


@app.callback(
    Output("dynamic_slider_container", "children"),
    Input("dropdown_slider", "value"),
    State("dynamic_slider_container", "children"),
    State({"type": "dynamic_slider", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def add_or_remove_slider(
    dropdown_names,
    children,
    current_ids,
):

    id_type = "dbc"

    current_names = get_current_names(current_ids)

    # add new slider
    if slider_is_added(current_names, dropdown_names):
        column = get_name_of_added_slider(current_names, dropdown_names)
        new_element = make_new_slider(column, id_type=id_type)
        children.append(new_element)

    # remove selected slider
    else:
        column = get_name_of_removed_slider(current_names, dropdown_names)
        remove_name_from_children(column, children, id_type=id_type)

    return children


@app.callback(
    Output({"type": "dynamic_slider_name", "index": MATCH}, "children"),
    Input({"type": "dynamic_slider", "index": MATCH}, "value"),
    State({"type": "dynamic_slider", "index": MATCH}, "id"),
    prevent_initial_call=True,
)
def update_slider_name(dynamic_slider_values, dynamic_slider_name):
    column = dynamic_slider_name["index"]
    name = get_slider_name(column, dynamic_slider_values)
    return name


#%%


@app.callback(
    Output("tax_id_filter_counts_output", "children"),
    Input("tax_id_filter_input_descendants", "value"),
    Input("tax_id_filter_subspecies", "value"),
)
def update_tax_id_filter_counts(tax_name, subspecies):

    if tax_name is None or tax_name == "":
        return f"No specific Tax IDs selected, defaults to ALL."
        # raise PreventUpdate

    tax_ids = taxonomy.extract_descendant_tax_ids(
        tax_name,
        include_subspecies=include_subspecies(subspecies),
    )
    N_tax_ids = len(tax_ids)
    if N_tax_ids == 0:
        return f"Couldn't find any Tax IDs for {tax_name} in NCBI"
    return f"Found {utils.human_format(N_tax_ids)} Tax IDs for {tax_name} in NCBI"


#%%


@app.callback(
    Output("filters_dropdown_files", "is_open"),
    Output("filters_toggle_files_button", "outline"),
    Input("filters_toggle_files_button", "n_clicks"),
    State("filters_dropdown_files", "is_open"),
)
def toggle_collapse_files(n, is_open):
    # after click
    if n:
        return not is_open, is_open
    # initial setup
    return is_open, True


@app.callback(
    Output("filters_dropdown_tax_ids", "is_open"),
    Output("filters_toggle_tax_ids_button", "outline"),
    Input("filters_toggle_tax_ids_button", "n_clicks"),
    State("filters_dropdown_tax_ids", "is_open"),
)
def toggle_collapse_tax_ids(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


@app.callback(
    Output("filters_dropdown_ranges_button", "is_open"),
    Output("filters_toggle_ranges_button", "outline"),
    Input("filters_toggle_ranges_button", "n_clicks"),
    State("filters_dropdown_ranges_button", "is_open"),
)
def toggle_collapse_ranges(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


#%%


@app.callback(
    Output("content_main", "style"),
    Output("sidebar_filter", "style"),
    Output("sidebar_plot", "style"),
    Output("sidebar_filter_state", "data"),
    Output("sidebar_plot_state", "data"),
    Input("btn_toggle_filter", "n_clicks"),
    Input("btn_toggle_plot", "n_clicks"),
    State("sidebar_filter_state", "data"),
    State("sidebar_plot_state", "data"),
)
def toggle_sidebars(
    _btn_toggle_filter,
    _btn_toggle_plot,
    current_state_sidebar_filter,
    current_state_sidebar_plot,
):

    button_id = dashboard_helper.get_button_id(dash.callback_context)

    # if the toggle filter button was clicked
    if button_id == "btn_toggle_filter":
        return dashboard_helper.toggle_filter(
            configurations,
            current_state_sidebar_filter,
            current_state_sidebar_plot,
        )

    # if the toggle plot button was clicked
    elif button_id == "btn_toggle_plot":
        return dashboard_helper.toggle_plot(
            configurations,
            current_state_sidebar_filter,
            current_state_sidebar_plot,
        )

    # base configuration
    else:
        return start_configuration


if __name__ == "__main__":
    app.run_server(debug=True)
