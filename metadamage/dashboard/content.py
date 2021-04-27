# Standard Library
import itertools

# Third Party
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

# First Party
from metadamage import dashboard, utils


#%%

d_columns_latex, columns, columns_no_log = dashboard.utils.get_d_columns_latex()

#%%


def get_navbar():

    navbar = dbc.NavbarSimple(
        [
            dbc.Button(
                "Filters",
                outline=True,
                color="light",
                className="mr-1",
                id="sidebar_left_toggle_btn",
            ),
            dbc.Button(
                "Counts",
                outline=True,
                color="light",
                className="mr-1",
                id="sidebar_right_toggle_btn",
            ),
            dbc.Button(
                "Styling",
                outline=True,
                color="light",
                className="mr-1",
                id="navbar_btn_toggle_styling",
            ),
        ],
        brand="mDamage",
        brand_href="#",
        color="dark",
        dark=True,
        fluid=True,
    )

    return navbar


#%%


def get_content_main(start_configuration):

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
                dcc.Graph(id="main_graph", **dashboard.utils.get_graph_kwargs()),
                dbc.Collapse(
                    [
                        dbc.Row(dbc.Col(html.Hr())),
                        dbc.Row(
                            dbc.Col(html.Center("Axis variables", className="lead"))
                        ),
                        dbc.Row(XY_axis_dropdowns, no_gutters=True),
                        dbc.Row(dbc.Col(html.Hr())),
                        dbc.Row(dbc.Col(html.Center("Marker size", className="lead"))),
                        dbc.Row(marker_transformations, no_gutters=True),
                    ],
                    id="navbar_collapsed_toggle_styling",
                    is_open=False,
                ),
            ]
        ),
        id="content_main",
        style=start_configuration.style_content_main,
    )

    return content_main


#%%


#%%


def get_sidebar_left(fit_results, start_configuration):

    filter_dropdown_file = dbc.FormGroup(
        [
            dashboard.utils.get_dropdown_file_selection(
                fit_results=fit_results,
                id="sidebar_left_dropdown_samples",
                shortnames_to_show="each",  # one for each first letter in shortname
            ),
        ]
    )

    filters_collapse_files = html.Div(
        [
            dbc.Button(
                "Samples",
                id="sidebar_left_samples_btn",
                color="secondary",
                block=True,
                outline=True,
                size="lg",
            ),
            dbc.Collapse(
                filter_dropdown_file,
                id="sidebar_left_samples_collapsed",
                is_open=False,
            ),
        ]
    )

    filter_tax_id = dbc.Row(
        [
            dbc.Col(html.Br(), width=12),
            dbc.Col(html.H6("Specific taxas:"), width=12),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dcc.Dropdown(
                            id="sidebar_left_tax_id_input",
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
                            id="sidebar_left_tax_id_input_descendants",
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
                            id="sidebar_left_tax_id_subspecies",
                        ),
                    ]
                ),
                width=12,
            ),
            dbc.Col(html.P(id="sidebar_left_tax_id_counts_output"), width=12),
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
                id="sidebar_left_taxanomics_btn",
                color="secondary",
                block=True,
                outline=True,
                size="lg",
            ),
            dbc.Collapse(
                filter_tax_id,
                id="sidebar_left_taxanomics_collapsed",
                is_open=False,
            ),
        ]
    )

    #%%

    filters_collapse_ranges = html.Div(
        [
            dbc.Button(
                "Fits",
                id="sidebar_left_fit_results_btn",
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
                            id="sidebar_left_fit_results",
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
                        id="sidebar_left_fit_results_container",
                        children=[],
                        width=12,
                    ),
                ],
                id="sidebar_left_fit_results_collapsed",
                is_open=False,
            ),
        ]
    )

    sidebar_left = html.Div(
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
        id="sidebar_left",
        style=start_configuration.style_sidebar_left,
    )

    return sidebar_left


#%%


def get_sidebar_right(start_configuration):

    sidebar_right_fit_results = html.Div(
        [
            dbc.Button(
                "Fit Results",
                id="sidebar_right_btn_toggle_fit_results",
                color="secondary",
                block=True,
                outline=True,
                size="lg",
            ),
            dbc.Collapse(
                html.Div(
                    "sidebar_right_datatable_fit_results",
                    id="sidebar_right_datatable_fit_results",
                ),
                id="sidebar_right_collapsed_toggle_fit_results",
                is_open=False,
            ),
        ]
    )

    sidebar_right_collapsed_toggle_combined_graph = dbc.FormGroup(
        [
            dcc.Graph(
                figure=dashboard.figures.create_empty_figure(),
                id="sidebar_right_graph_combined",
                **dashboard.utils.get_graph_kwargs_no_buttons(),
            ),
        ]
    )

    sidebar_right_collapsed_toggle_combined = html.Div(
        [
            dbc.Button(
                "Combined",
                id="sidebar_right_btn_toggle_combined",
                color="secondary",
                block=True,
                outline=False,
                size="lg",
            ),
            dbc.Collapse(
                sidebar_right_collapsed_toggle_combined_graph,
                id="sidebar_right_collapsed_toggle_combined",
                is_open=True,
            ),
        ]
    )

    sidebar_right_collapsed_toggle_forward_reverse_graph = dbc.FormGroup(
        [
            dcc.Graph(
                figure=dashboard.figures.create_empty_figure(),
                id="sidebar_right_graph_forward",
                style={"height": "20vh"},
                **dashboard.utils.get_graph_kwargs_no_buttons(),
            ),
            dcc.Graph(
                figure=dashboard.figures.create_empty_figure(),
                id="sidebar_right_graph_reverse",
                style={"height": "20vh"},
                **dashboard.utils.get_graph_kwargs_no_buttons(),
            ),
        ]
    )

    sidebar_right_collapsed_toggle_forward_reverse = html.Div(
        [
            dbc.Button(
                "Forward / Reverse",
                id="sidebar_right_btn_toggle_forward_reverse",
                color="secondary",
                block=True,
                outline=True,
                size="lg",
            ),
            dbc.Collapse(
                sidebar_right_collapsed_toggle_forward_reverse_graph,
                id="sidebar_right_collapsed_toggle_forward_reverse",
                is_open=False,
            ),
        ]
    )

    sidebar_right = html.Div(
        [
            html.H2("Counts", className="display-4"),
            # html.Hr(),
            sidebar_right_collapsed_toggle_combined,
            html.Hr(),
            sidebar_right_fit_results,
            html.Hr(),
            sidebar_right_collapsed_toggle_forward_reverse,
        ],
        id="sidebar_right",
        style=start_configuration.style_sidebar_right,
    )

    return sidebar_right


#%%


def get_app_layout(fit_results, start_configuration):

    navbar = get_navbar()
    content_main = get_content_main(start_configuration)
    sidebar_left = get_sidebar_left(fit_results, start_configuration)
    sidebar_right = get_sidebar_right(start_configuration)

    return html.Div(
        [
            # dcc.Store(id="store"),
            dcc.Store(id="sidebar_right_state"),
            dcc.Store(id="sidebar_left_state"),
            navbar,
            sidebar_left,
            content_main,
            sidebar_right,
            dbc.Modal(
                [
                    dbc.ModalHeader("Filtering Error"),
                    dbc.ModalBody(
                        "Too restrictive filtering, no points left to plot. "
                        "Please choose a less restrictive filtering."
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id="modal_close_button", className="ml-auto"
                        )
                    ),
                ],
                centered=True,
                id="modal",
            ),
        ],
    )


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


def make_new_slider(fit_results, column, id_type, N_steps=100):

    d_range_slider = dashboard.utils.get_range_slider_keywords(
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
                    id={
                        "type": "sidebar_left_fit_results_dynamic_name",
                        "index": column,
                    },
                ),
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.RangeSlider(
                        id={
                            "type": "sidebar_left_fit_results_dynamic",
                            "index": column,
                        },
                        **d_range_slider,
                    ),
                    width=12,
                ),
            ),
        ],
        id={"type": id_type, "index": column},
    )
