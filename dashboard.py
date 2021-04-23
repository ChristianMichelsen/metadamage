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
    title="mDamage",
    update_title="Updating...",
)

# to allow custom css
# app.scripts.config.serve_locally = True

# First Party
from metadamage import dashboard

dashboard.utils.set_custom_theme()
# reload(dashboard)


fit_results = dashboard.fit_results.FitResults(
    folder=Path("./data/out/"),
    use_memoization=True,
)

d_columns_latex = dashboard_helper.get_d_columns_latex()
columns = list(d_columns_latex.keys())
columns_no_log = [col for col in columns if not col.startswith("log_")]

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
        # html.H4("Toggle", style={"textAlign": "center", "color": "white"}),
        dbc.Button(
            "Filters",
            outline=True,
            color="light",
            className="mr-1",
            id="btn_toggle_filter",
        ),
        dbc.Button(
            "Counts",
            outline=True,
            color="light",
            className="mr-1",
            id="btn_toggle_plot",
        ),
        dbc.Button(
            "Styling",
            outline=True,
            color="light",
            className="mr-1",
            id="btn_toggle_variables",
        ),
    ],
    brand="mDamage",
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

dropdown_y_axis = dcc.Dropdown(
    id="yaxis_column",
    options=[{"label": i, "value": i} for i in columns],
    value="D_max",
)


XY_axis_dropdowns = [
    dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(html.Center("X-axis: ")),
                    dbc.Col(dropdown_x_axis, width=12),
                ]
            ),
        ],
        width=6,
    ),
    dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(html.Center("Y-axis: ")),
                    dbc.Col(dropdown_y_axis, width=12),
                ]
            ),
        ],
        width=6,
    ),
]


marker_transformation_function = dcc.Dropdown(
    id="marker_transformation_function",
    options=[
        {"label": "Constant", "value": "constant"},
        {"label": "Identity", "value": "identity"},
        {"label": "Sqrt", "value": "sqrt"},
        {"label": "Log", "value": "log10"},
    ],
    value="sqrt",
    searchable=False,
    clearable=False,
)


marker_transformation_variable = dcc.Dropdown(
    id="marker_transformation_variable",
    options=[{"label": col, "value": col} for col in columns_no_log],
    value="N_alignments",
    searchable=True,
    clearable=False,
)

marker_transformation_slider = dcc.Slider(
    id="marker_transformation_slider",
    min=1,
    max=60,
    step=1,
    value=30,
    marks={mark: str(mark) for mark in [1, 10, 20, 30, 40, 50, 60]},
)

marker_transformations = [
    dbc.Col(
        dbc.Row(
            [
                dbc.Col(html.Center("Variable:")),
                dbc.Col(marker_transformation_variable, width=12),
            ],
        ),
        width=4,
    ),
    dbc.Col(
        dbc.Row(
            [
                dbc.Col(html.Center("Function:")),
                dbc.Col(marker_transformation_function, width=12),
            ],
        ),
        width=2,
    ),
    dbc.Col(
        dbc.Row(
            [
                dbc.Col(html.Center("Scale:")),
                dbc.Col(marker_transformation_slider, width=12),
            ],
        ),
        width=6,
    ),
]


content_main = html.Div(
    html.Div(
        [
            dcc.Graph(id="indicator_graphic", **dashboard_helper.get_graph_kwargs()),
            dbc.Collapse(
                [
                    dbc.Row(dbc.Col(html.Hr())),
                    dbc.Row(dbc.Col(html.Center("Axis variables", className="lead"))),
                    dbc.Row(XY_axis_dropdowns, no_gutters=True),
                    dbc.Row(dbc.Col(html.Hr())),
                    dbc.Row(dbc.Col(html.Center("Marker size", className="lead"))),
                    dbc.Row(marker_transformations, no_gutters=True),
                ],
                id="collapsed_variable_selections",
                is_open=False,
            ),
        ]
    ),
    id="content_main",
    style=start_configuration.style_content_main,
)


@app.callback(
    Output("collapsed_variable_selections", "is_open"),
    Output("btn_toggle_variables", "outline"),
    Input("btn_toggle_variables", "n_clicks"),
    State("collapsed_variable_selections", "is_open"),
)
def toggle_collapse_files(n, is_open):
    # after click
    if n:
        return not is_open, is_open
    # initial setup
    return is_open, True


#%%


filter_dropdown_file = dbc.FormGroup(
    [
        # html.Br(),
        # dbc.Col(html.H3("Samples"), width=12),
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
            "Samples",
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
        dbc.Col(html.H6("Specific taxas:"), width=12),
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
        dbc.Col(html.H6("Taxanomic descendants:"), width=12),
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
                        "Update",
                        id="tax_id_plot_button",
                        color="light",
                        block=True,
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
            "Taxanomics",
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


filters_collapse_ranges = html.Div(
    [
        dbc.Button(
            "Fits",
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
                    html.H6("Fit results:"),
                    width=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown_slider",
                        options=[
                            {"label": shortname, "value": shortname}
                            for shortname in columns_no_log
                        ],
                        value=[],
                        multi=True,
                        placeholder="Select a variable ...",
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
        html.H2("Filters", className="display-4"),
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


sidebar_plot_fit_results = html.Div(
    [
        dbc.Button(
            "Fit Results",
            id="sidebar_plot_toggle_fit_results",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            html.Div("Blabla", id="blabla"),
            id="sidebar_plot_fit_results",
            is_open=False,
        ),
    ]
)


@app.callback(
    Output("sidebar_plot_fit_results", "is_open"),
    Output("sidebar_plot_toggle_fit_results", "outline"),
    Input("sidebar_plot_toggle_fit_results", "n_clicks"),
    State("sidebar_plot_fit_results", "is_open"),
)
def toggle_collapse_plot_combined(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


sidebar_plot_combined_graph = dbc.FormGroup(
    [
        dcc.Graph(
            figure=dashboard.figures.create_empty_figure(),
            id="graph_plot_data",
            **dashboard_helper.get_graph_kwargs_no_buttons(),
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
            **dashboard_helper.get_graph_kwargs_no_buttons(),
        ),
        dcc.Graph(
            figure=dashboard.figures.create_empty_figure(),
            id="graph_plot_data_reverse",
            style={"height": "20vh"},
            **dashboard_helper.get_graph_kwargs_no_buttons(),
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
        html.H2("Counts", className="display-4"),
        # html.Hr(),
        sidebar_plot_combined,
        html.Hr(),
        sidebar_plot_fit_results,
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
        "k",
        "N",
    ]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
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


def update_raw_count_plots(click_data, forward_reverse):
    if click_data is not None:

        shortname, tax_id = get_shortname_tax_id_from_click_data(
            fit_results,
            click_data,
        )
        group = fit_results.get_single_count_group(
            shortname,
            tax_id,
            forward_reverse,
        )
        fit = fit_results.get_single_fit_prediction(
            shortname,
            tax_id,
            forward_reverse,
        )
        fig = plot_group(group, fit, forward_reverse)
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output("graph_plot_data", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_raw_count_plots_combined(click_data):
    return update_raw_count_plots(click_data, forward_reverse="")


@app.callback(
    Output("graph_plot_data_forward", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_raw_count_plots_forward(click_data):
    return update_raw_count_plots(click_data, forward_reverse="Forward")


@app.callback(
    Output("graph_plot_data_reverse", "figure"),
    Input("indicator_graphic", "clickData"),
)
def update_raw_count_plots_reverse(click_data):
    return update_raw_count_plots(click_data, forward_reverse="Reverse")


import dash_table


@app.callback(
    Output("blabla", "children"),
    Input("indicator_graphic", "clickData"),
)
def update_datatable(click_data):
    if click_data:
        shortname, tax_id = get_shortname_tax_id_from_click_data(
            fit_results, click_data
        )

        df_fit = fit_results.filter({"shortname": shortname, "tax_id": tax_id})
        if len(df_fit) != 1:
            raise AssertionError(f"Should only be length 1")

        ds = df_fit.iloc[0]

        # fmt: off
        lines = [
            html.Br(),
            f"Name: {ds['shortname']}", html.Br(), html.Br(),

            f"Tax Name: {ds['tax_name']}", html.Br(),
            f"Tax Rank: {ds['tax_rank']}", html.Br(),
            f"Tax ID: {ds['tax_id']}", html.Br(), html.Br(),

            f"LR: {ds['LR']:.2f}",html.Br(),
            f"D_max: {ds['D_max']:.3f} ± {ds['D_max_std']:.3f}", html.Br(),
            f"q: {ds['q']:.3f} ± {ds['q_std']:.3f}", html.Br(),
            f"phi: {ds['phi']:.1f} ± {ds['phi_std']:.1f}", html.Br(),
            f"asymmetry: {ds['asymmetry']:.3f}", html.Br(),
            f"rho_Ac: {ds['rho_Ac']:.3f}", html.Br(),
        ]
        # fmt: on

        return html.P(lines)

    return html.P(["Please select a point"])


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


def make_figure(df, xaxis_column_name, yaxis_column_name):

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

    # 2. * max(array of size values) / (desired maximum marker size ** 2)

    fig.update_traces(
        hovertemplate=fit_results.hovertemplate,
        marker_line_width=0,
        marker_sizemode="area",
        marker_sizeref=2.0 * fit_results.max_of_size / (fit_results.marker_size ** 2),
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

    fig.update_xaxes(title=d_columns_latex[xaxis_column_name])
    fig.update_yaxes(title=d_columns_latex[yaxis_column_name])

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
    Input("marker_transformation_variable", "value"),
    Input("marker_transformation_function", "value"),
    Input("marker_transformation_slider", "value"),
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
    marker_transformation_variable,
    marker_transformation_function,
    marker_transformation_slider,
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

    # print(
    #     marker_transformation_variable,
    #     marker_transformation_function,
    #     marker_transformation_slider,
    # )

    fit_results.set_marker_size(
        marker_transformation_variable,
        marker_transformation_function,
        marker_transformation_slider,
    )

    d_filter = {"shortnames": dropdown_file_selection}

    columns_no_log = [id["index"] for id in slider_ids]
    for shortname, values in zip(columns_no_log, slider_values):
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

    if dashboard.utils.is_log_transform_column(column):
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
def add_or_remove_slider(dropdown_names, children, current_ids):

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
    Output("btn_toggle_filter", "outline"),
    Output("btn_toggle_plot", "outline"),
    Input("btn_toggle_filter", "n_clicks"),
    Input("btn_toggle_plot", "n_clicks"),
    State("sidebar_filter_state", "data"),
    State("sidebar_plot_state", "data"),
    State("btn_toggle_filter", "outline"),
    State("btn_toggle_plot", "outline"),
)
def toggle_sidebars(
    _btn_toggle_filter,
    _btn_toggle_plot,
    current_state_sidebar_filter,
    current_state_sidebar_plot,
    btn_toggle_filter_outline,
    btn_toggle_plot_outline,
):

    button_id = dashboard_helper.get_button_id(dash.callback_context)

    # if the toggle filter button was clicked
    if button_id == "btn_toggle_filter":
        return (
            *dashboard_helper.toggle_filter(
                configurations,
                current_state_sidebar_filter,
                current_state_sidebar_plot,
            ),
            not btn_toggle_filter_outline,
            btn_toggle_plot_outline,
        )

    # if the toggle plot button was clicked
    elif button_id == "btn_toggle_plot":
        return (
            *dashboard_helper.toggle_plot(
                configurations,
                current_state_sidebar_filter,
                current_state_sidebar_plot,
            ),
            btn_toggle_filter_outline,
            not btn_toggle_plot_outline,
        )

    # base configuration
    else:
        return *start_configuration, True, True


if __name__ == "__main__":
    app.run_server(debug=True)
