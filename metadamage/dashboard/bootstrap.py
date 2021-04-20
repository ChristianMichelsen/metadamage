# Third Party
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

# First Party
from metadamage import dashboard


class Bootstrap:
    def __init__(self, fit_results, graph_kwargs):
        self.fit_results = fit_results

        tabs_kwargs = [
            dict(label="Overview", tab_id="fig_fit_results"),
            dict(label="Histograms", tab_id="fig_histograms"),
            dict(label="Scatter Matrix", tab_id="fig_scatter_matrix"),
            dict(label="Forward / Reverse", tab_id="fig_forward_reverse"),
        ]

        tabs = dbc.Tabs(
            [dbc.Tab(**tab) for tab in tabs_kwargs],
            id="tabs",
            active_tab="fig_fit_results",
        )

        self.card_tabs = dbc.Card(
            [tabs],
            body=True,
        )

        #%%

        mismatch_dropdown_shortnames = dbc.FormGroup(
            [
                dbc.Label("Filenames:", className="mr-2"),
                html.Div(
                    dcc.Dropdown(
                        id="dropdown_mismatch_shortname",
                        options=[
                            {"label": shortname, "value": shortname}
                            for shortname in fit_results.shortnames
                        ],
                        clearable=False,
                    ),
                    style={"min-width": "300px"},
                ),
            ],
            className="mr-5",
        )

        mismatch_dropdown_tax_ids = dbc.FormGroup(
            [
                dbc.Label("Tax IDs:", className="mr-2"),
                html.Div(
                    dcc.Dropdown(id="dropdown_mismatch_tax_id", clearable=True),
                    style={"min-width": "150px"},
                ),
            ],
            className="mr-3",
        )

        mismatch_dropdowns = dbc.Card(
            [
                dbc.Form(
                    [
                        mismatch_dropdown_shortnames,
                        mismatch_dropdown_tax_ids,
                    ],
                    inline=True,
                ),
            ],
            body=True,
        )

        self.card_mismatch_dropdowns_and_graph = dbc.Card(
            [
                dbc.Row(
                    dbc.Col(mismatch_dropdowns, width=10),
                    justify="center",
                ),
                dcc.Graph(
                    figure=dashboard.figures.create_empty_figure(),
                    id="graph_mismatch",
                    **graph_kwargs,
                ),
            ],
            body=True,
        )

        #%%

        self.card_datatable = dbc.Card(
            [
                html.H3("Datatable", className="card-title"),
                DataTable(
                    **dashboard.datatable.get_data_table_keywords(id="data_table")
                ),
            ],
            body=True,
        )

        #%%

        # Standard Library
        import itertools

        filter_tax_id = dbc.Row(
            [
                dbc.Col(
                    html.Br(),
                    width=12,
                ),
                dbc.Col(
                    html.H3("Filter specific taxa"),
                    width=12,
                ),
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
                dbc.Col(
                    html.Br(),
                    width=12,
                ),
                dbc.Col(
                    html.H3("Filter taxanomic descendants"),
                    width=12,
                ),
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
                dbc.Col(
                    html.P(id="tax_id_filter_counts_output"),
                    # xl=9,
                    width=12,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Button(
                                "Plot",
                                id="tax_id_plot_button",
                                color="primary",
                                block=True,
                            ),
                        ]
                    ),
                    # xl=3,
                    # offset=3,
                    # width=6,
                    xs={"size": 11, "offset": 1},
                    sm={"size": 11, "offset": 1},
                    md={"size": 10, "offset": 2},
                    lg={"size": 9, "offset": 3},
                    width={"size": 6, "offset": 6},
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

        filter_dropdown_file = dbc.FormGroup(
            [
                # dbc.Label("Dropdown"),
                html.Br(),
                dbc.Col(
                    html.H3("Filter input samples"),
                    width=12,
                ),
                dashboard.elements.get_dropdown_file_selection(
                    id="dropdown_file_selection",
                    fit_results=fit_results,
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

        slider_names = [
            "n_sigma",
            "D_max",
            "q_mean",
            "concentration_mean",
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
                            html.H3("Filter fit results"),
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
                            id="dynamic_slider-container",
                            children=[],
                            width=12,
                        ),
                    ],
                    id="filters_dropdown_ranges_button",
                    is_open=False,
                ),
            ]
        )

        self.filter_card = dbc.Card(
            [
                html.H1("Select filters:", className="card-title"),
                html.Hr(),
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
            # style={"maxHeight": "1000px", "overflow": "auto"},
            body=True,  # spacing before border
        )

        #%%

        form_overview_marker_size = dbc.FormGroup(
            [
                dbc.Label("Marker scaling", className="mr-2"),
                # dbc.Col(
                html.Div(
                    dcc.Slider(  # possible fix for ReferenceError
                        id={"type": "slider_overview_marker_size", "index": 0},
                        **dashboard.elements.get_slider_keywords(),
                    ),
                    style={"min-width": "300px"},
                )
                # width=5,
                # ),
            ],
            className="mr-6",
        )

        form_overview_marker_transformation = dbc.FormGroup(
            [
                dbc.Label("Marker transformation", className="mr-2"),
                # dbc.Col(
                dcc.Dropdown(  # possible fix for ReferenceError
                    id={"type": "dropdown_overview_marker_transformation", "index": 0},
                    options=[
                        {"label": "Identity", "value": "identity"},
                        {"label": "Sqrt", "value": "sqrt"},
                        {"label": "Log", "value": "log10"},
                        {"label": "Constant", "value": "constant"},
                    ],
                    value="sqrt",
                    searchable=False,
                    clearable=False,
                    style={"min-width": "100px"},
                    # ),
                    # width=5,
                ),
            ],
            className="mr-6",
        )

        button_reset = dbc.FormGroup(
            [
                dbc.Label("", className="mr-5"),
                dbc.Button(
                    "Reset",
                    id={"type": "button_reset", "index": 0},
                    color="primary",
                ),
            ],
            className="mr-6",
        )

        self.card_overview_marker = dbc.Card(
            [
                # html.H3("Markers", className="card-title"),
                dbc.Form(
                    [
                        form_overview_marker_size,
                        form_overview_marker_transformation,
                        button_reset,
                    ],
                    inline=True,
                ),
            ],
            body=True,  # spacing before border
            # className="mr-6",
        )
