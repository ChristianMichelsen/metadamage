# Scientific Library
import numpy as np
import pandas as pd

# Third Party
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

# First Party
# import dashboard
from metadamage import dashboard
from metadamage.utils import human_format


def get_shortnames_each(all_shortnames):
    first_letters = {s[0] for s in all_shortnames}
    values = []
    for first_letter in first_letters:
        for shortname in all_shortnames:
            if shortname[0] == first_letter:
                values.append(shortname)
                break
    return values


def get_dropdown_file_selection(fit_results, id, shortnames_to_show="all"):

    special_shortnames = ["Select all", "Default selection"]
    N_special_shortnames = len(special_shortnames)
    all_shortnames = special_shortnames + fit_results.shortnames

    if shortnames_to_show is None:
        values = all_shortnames

    elif isinstance(shortnames_to_show, int):
        values = all_shortnames[: shortnames_to_show + N_special_shortnames]

    elif isinstance(shortnames_to_show, str):

        if shortnames_to_show == "all":
            values = all_shortnames

        elif shortnames_to_show == "each":
            values = get_shortnames_each(fit_results.shortnames)

    values = list(sorted(values))

    dropdown_file_selection = dcc.Dropdown(
        id=id,
        options=[
            {"label": shortname, "value": shortname} for shortname in all_shortnames
        ],
        value=values,
        multi=True,
        placeholder="Select files to plot",
    )

    return dropdown_file_selection


#%%


def _insert_mark_values(mark_values):
    # https://github.com/plotly/dash-core-components/issues/159
    # work-around bug reported in https://github.com/plotly/dash-core-components/issues/159
    # if mark keys happen to fall on integers, cast them to int

    mark_labels = {}
    for mark_val in mark_values:
        # close enough to an int for my use case
        if abs(mark_val - round(mark_val)) < 1e-3:
            mark_val = int(mark_val)
        mark_labels[mark_val] = human_format(mark_val)
    return mark_labels


def get_range_slider_keywords(fit_results, column="N_alignments", N_steps=100):

    no_min = "Min"
    no_max = "Max"

    df = fit_results.df_fit_results

    if dashboard.utils.is_log_transform_column(column):
        # if column in dashboard.utils.log_transform_columns:

        x = df[column]

        range_log = np.log10(x[x > 0])
        range_min = np.floor(range_log.min())
        range_max = np.ceil(range_log.max())
        marks_steps = np.arange(range_min, range_max + 1)

        # if x contains 0-values
        if (x <= 0).sum() != 0:
            range_min = -1
            marks_steps = np.insert(marks_steps, 0, -1)

        if len(marks_steps) > 6:
            marks_steps = (
                [marks_steps[0]] + [x for x in marks_steps[1:-1:2]] + [marks_steps[-1]]
            )

        f = lambda x: human_format(dashboard.utils.log_transform_slider(x))
        marks = {int(i): f"{f(i)}" for i in marks_steps}

        marks[marks_steps[0]] = {"label": no_min, "style": {"color": "#a3ada9"}}
        marks[marks_steps[-1]] = {"label": no_max, "style": {"color": "#a3ada9"}}

    elif column == "D_max" or column == "q" or column == "c":
        range_min = 0.0
        range_max = 1.0
        marks = {
            0.25: "0.25",
            0.5: "0.5",
            0.75: "0.75",
        }
        marks[0] = {"label": no_min, "style": {"color": "#a3ada9"}}
        marks[1] = {"label": no_max, "style": {"color": "#a3ada9"}}

    else:

        array = df[column]
        array = array[np.isfinite(array) & array.notnull()]

        range_min = np.min(array)
        range_max = np.max(array)

        if range_max - range_min > 1:
            range_min = np.floor(range_min)
            range_max = np.ceil(range_max)
            mark_values = np.linspace(range_min, range_max, 5, dtype=int)
            marks = _insert_mark_values(mark_values[1:-1])

        else:
            decimals = abs(int(np.floor(np.log10(range_max - range_min))))
            range_min = np.around(range_min, decimals=decimals)
            range_max = np.around(range_max, decimals=decimals)

            mark_values = np.linspace(range_min, range_max, 5)
            marks = {float(val): str(val) for val in mark_values[1:-1]}

        marks[int(mark_values[0])] = {"label": no_min, "style": {"color": "#a3ada9"}}
        marks[int(mark_values[-1])] = {
            "label": no_max,
            "style": {"color": "#a3ada9"},
        }

    step = (range_max - range_min) / N_steps

    return dict(
        min=range_min,
        max=range_max,
        step=step,
        marks=marks,
        value=[range_min, range_max],
        allowCross=False,
        updatemode="mouseup",
        included=True,
        # tooltip=dict(
        #     always_visible=False,
        #     placement="bottom",
        # ),
    )


#%%


def get_card_D_max_slider(fit_results):

    card_D_max_slider = dbc.Card(
        html.Div(
            [
                dbc.Row(
                    dcc.Markdown(id="range-slider-D-max-output"),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.RangeSlider(
                            id="range-slider-D-max",
                            **get_range_slider_keywords(
                                fit_results.df,
                                column="D_max",
                                N_steps=100,
                            ),
                        ),
                    ),
                    justify="center",
                ),
            ],
            style={
                "marginBottom": "1em",
                "marginTop": "0.5em",
                "marginLeft": "0.7em",
                "marginRight": "0.7em",
            },
        ),
        outline=True,
        color="white",
        # className="w-100",
    )

    return card_D_max_slider


def get_card_N_alignments_slider(fit_results):

    card_N_alignments_slider = dbc.Card(
        html.Div(
            [
                dbc.Row(
                    dcc.Markdown(id="range-slider-N-alignments-output"),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.RangeSlider(
                            id="range-slider-N-alignments",
                            **get_range_slider_keywords(
                                fit_results.df,
                                column="N_alignments",
                                N_steps=100,
                            ),
                        ),
                    ),
                    justify="center",
                ),
            ],
            style={
                "marginBottom": "1em",
                "marginTop": "0.5em",
                "marginLeft": "0.7em",
                "marginRight": "0.7em",
            },
        ),
        outline=True,
        color="white",
    )

    return card_N_alignments_slider


def get_card_dropdown_file_selection(fit_results):

    card_dropdown_file_selection = dbc.Card(
        [
            html.H3("File Selection", className="card-title"),
            get_dropdown_file_selection(fit_results=fit_results, id="XXX"),
        ],
        body=True,  # spacing before border
    )
    return card_dropdown_file_selection


def get_card_filters(fit_results):

    card_filters = dbc.Card(
        [
            html.H3("Filters", className="card-title"),
            get_card_N_alignments_slider(fit_results),
            get_card_D_max_slider(fit_results),
        ],
        body=True,  # spacing before border
    )

    return card_filters


def get_card_datatable(fit_results):

    card_datatable = dbc.Card(
        [
            dbc.Row(
                [
                    # dbc.Col(html.H3("Table", className="card-title"), md=1),
                    dbc.Col(
                        DataTable(**dashboard.datatable.get_data_table_keywords()),
                        # md=10,
                    ),
                ],
            )
        ],
        body=True,  # spacing before border
    )

    return card_datatable


def get_dropdown_marker_transformation(fit_results):

    dropdown_marker_transformation = dcc.Dropdown(
        id="dropdown_marker_transformation",
        options=[
            {"label": "Identity", "value": "identity"},
            {"label": "Sqrt", "value": "sqrt"},
            {"label": "Log", "value": "log10"},
            {"label": "Constant", "value": "constant"},
        ],
        value="sqrt",
        searchable=False,
        clearable=False,
    )

    return dropdown_marker_transformation


def get_slider_keywords():
    marks = [1, 10, 20, 30, 40, 50, 60]
    return dict()


def get_card_marker_size_slider(fit_results):

    card_marker_size_slider = dbc.Card(
        html.Div(
            [
                dbc.Row(
                    dcc.Markdown(id="slider-marker-size-output"),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Slider(
                            id="slider-marker-size",
                            **get_slider_keywords(),
                        ),
                    ),
                    justify="center",
                ),
            ],
            style={
                "marginBottom": "1em",
                "marginTop": "0.5em",
                "marginLeft": "0.7em",
                "marginRight": "0.7em",
            },
        ),
        outline=True,
        color="white",
        # className="w-100",
    )

    return card_marker_size_slider


def get_card_dropdown_marker_transformation(fit_results):

    card_dropdown_marker_transformation = dbc.Card(
        [
            html.H3("Marker Size", className="card-title"),
            get_card_marker_size_slider(fit_results),
            get_dropdown_marker_transformation(fit_results),
        ],
        body=True,  # spacing before border
    )
    return card_dropdown_marker_transformation


def get_card_filter_and_dropdown_file_selection(fit_results):

    card_filter_and_dropdown_file_selection = dbc.Card(
        [
            html.Br(),
            get_card_dropdown_file_selection(fit_results),
            html.Br(),
            get_card_filters(fit_results),
            html.Br(),
            get_card_dropdown_marker_transformation(fit_results),
        ],
        body=True,  # spacing before border
        outline=True,  # together with color, makes a transparent/white border
        color="white",
    )
    return card_filter_and_dropdown_file_selection


def get_card_main_plot(fit_results):

    card_main_plot = dbc.Card(
        [
            html.Br(),
            # this has to be a div for "reset axis" to work properly
            html.Div(id="tab_figure"),
        ],
        body=True,  # spacing before border
        outline=True,  # together with color, makes a transparent/white border
        color="white",
    )

    return card_main_plot


def get_card_mismatch_figure(fit_results):

    card_mismatch_figure = dbc.Card(
        [
            html.Br(),
            # dcc.Graph(id="mismatch_figure")
            html.Div(id="mismatch_figure-div"),
            # this has to be a div for "reset axis" to work properly
            # html.Div(id="mismatch_figure_div"),
        ],
        body=True,  # spacing before border
        outline=True,  # together with color, makes a transparent/white border
        color="white",
    )

    return card_mismatch_figure
