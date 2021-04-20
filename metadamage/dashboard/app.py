# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from importlib import reload
from pathlib import Path
from threading import Timer
import time
import webbrowser

# Third Party
from about_time import about_time
import dash
from dash.dependencies import ALL, Input, MATCH, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# First Party
from metadamage import dashboard, io, taxonomy, utils


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
    # https://css-tricks.com/fun-viewport-units/
    style={"width": "100%", "height": "55vh"},
)


def get_app(out_dir_default, verbose=True):

    # Third Party
    # from metadamage import dashboard

    dashboard.utils.set_custom_theme()

    if verbose:
        print(f"Getting app now from {out_dir_default}")

    with about_time() as at1:
        fit_results = dashboard.fit_results.FitResults(
            folder=out_dir_default,
            verbose=verbose,
            very_verbose=False,
        )
    # print(f"{at1.duration_human}")

    bootstrap = dashboard.bootstrap.Bootstrap(fit_results, graph_kwargs)

    #%%BOOTSTRAP

    # external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.COSMO],
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
            "MathJax.js?config=TeX-MML-AM_CHTML",
        ],
        suppress_callback_exceptions=True,
        title="Metadamage",
        update_title="Updating...",
    )

    # to allow custom css
    app.scripts.config.serve_locally = True

    #%%

    def get_figure_from_df_and_tab(df_fit_results_filtered, active_tab):
        if active_tab == "fig_fit_results":
            fig = dashboard.figures.plot_fit_results(
                fit_results, df_fit_results_filtered
            )
        elif active_tab == "fig_histograms":
            fig = dashboard.figures.plot_histograms(
                fit_results, df_fit_results_filtered
            )
        elif active_tab == "fig_scatter_matrix":
            fig = dashboard.figures.plot_scatter_matrix(
                fit_results, df_fit_results_filtered
            )
        elif active_tab == "fig_forward_reverse":
            fig = dashboard.figures.plot_forward_reverse(
                fit_results, df_fit_results_filtered
            )
        else:
            print("got here: get_figure_from_df_and_tab")
        return fig

    def get_main_figure(data_or_df, active_tab="fig_fit_results"):
        if data_or_df is None:
            return dashboard.figures.create_empty_figure(s="")

        if isinstance(data_or_df, list):
            df = pd.DataFrame.from_records(data_or_df)
        elif isinstance(data_or_df, pd.DataFrame):
            df = data_or_df
        else:
            raise AssertionError(f"Got wrong type for data_or_df: {type(data_or_df)}")
        return get_figure_from_df_and_tab(df, active_tab)

    def make_tab_from_data(data_or_df=None, active_tab="fig_fit_results"):

        figure = get_main_figure(data_or_df, active_tab)
        main_graph = dcc.Graph(figure=figure, id="main_graph", **graph_kwargs)

        if active_tab == "fig_fit_results" or active_tab == "overview":

            return (
                dbc.Container(
                    [
                        dbc.Row(
                            dbc.Col(main_graph, width=12),
                            justify="center",
                        ),
                        dbc.Row(
                            dbc.Col(bootstrap.card_overview_marker, width=10),
                            justify="center",
                        ),
                    ],
                ),
            )

        elif active_tab == "fig_histograms":
            return (
                dbc.Container(
                    [
                        dbc.Row(
                            dbc.Col(main_graph, width=12),
                            justify="center",
                        ),
                    ],
                ),
            )

        elif active_tab == "fig_scatter_matrix":
            return (
                dbc.Container(
                    [
                        dbc.Row(
                            dbc.Col(main_graph, width=12),
                            justify="center",
                        ),
                    ],
                ),
            )

        elif active_tab == "fig_forward_reverse":
            return (
                dbc.Container(
                    [
                        dbc.Row(
                            dbc.Col(main_graph, width=12),
                            justify="center",
                        ),
                    ],
                ),
            )

        else:
            print("got here: make_tab_from_data")

    #%%

    card_graph = dbc.Card(
        [
            html.Div(
                make_tab_from_data(active_tab="overview"),  # main_graph
                id="main_graph_div",
                className="loader-fade",
            ),
        ],
        body=True,  # spacing before border
        # style={"height": "50vh"},
    )

    #%%

    app.layout = dbc.Container(
        [
            dcc.Store(id="store"),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(bootstrap.filter_card, width=3),
                    dbc.Col([bootstrap.card_tabs, card_graph], width=6),
                    dbc.Col(bootstrap.card_mismatch_dropdowns_and_graph, width=3),
                ],
                justify="center",
                # className="h-75",
            ),
            html.Hr(),
            dbc.Row(
                [dbc.Col(bootstrap.card_datatable, width=12)],
                justify="center",
                # className="h-25",
            ),
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
        fluid=True,  # fill available horizontal space and resize fluidly
        # style={"height": "90vh"},
    )

    #%%

    def key_is_in_list_case_insensitive(lst, key):
        return any([key.lower() in s.lower() for s in lst])

    @app.callback(
        Output("dropdown_file_selection", "value"),
        Input("dropdown_file_selection", "value"),
    )
    def update_dropdown_when_Select_All(dropdown_file_selection):
        if key_is_in_list_case_insensitive(dropdown_file_selection, "Select all"):
            dropdown_file_selection = fit_results.shortnames
        elif key_is_in_list_case_insensitive(dropdown_file_selection, "Deselect"):
            dropdown_file_selection = dashboard.elements.get_shortnames_each(
                fit_results.shortnames
            )
        return dropdown_file_selection

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

    @app.callback(
        Output("store", "data"),
        Output("modal", "is_open"),
        Input("dropdown_file_selection", "value"),
        Input("tax_id_filter_input", "value"),
        Input("tax_id_plot_button", "n_clicks"),
        Input({"type": "dynamic_slider", "index": ALL}, "value"),
        Input({"type": "slider_overview_marker_size", "index": ALL}, "value"),
        Input(
            {"type": "dropdown_overview_marker_transformation", "index": ALL}, "value"
        ),
        Input("modal_close_button", "n_clicks"),
        State("tax_id_filter_input_descendants", "value"),
        State("tax_id_filter_subspecies", "value"),
        State({"type": "dynamic_slider", "index": ALL}, "id"),
        State("modal", "is_open"),
        # State("tabs", "active_tab"),
    )
    def filter_fit_results(
        dropdown_file_selection,
        tax_id_filter_input,
        tax_id_button,
        slider_values,
        marker_size_max,
        marker_transformation,
        n_clicks_modal,
        tax_id_filter_input_descendants,
        tax_id_filter_subspecies,
        slider_ids,
        modal_is_open,
        # active_tab,
        prevent_initial_call=True,
    ):

        # if modal is open and the "close" button is clicked, close down modal
        if n_clicks_modal and modal_is_open:
            return dash.no_update, False

        # if no files selected
        if not dropdown_file_selection:
            raise PreventUpdate

        with about_time() as at1:

            fit_results.set_marker_size(marker_transformation, marker_size_max)

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

        # print(f"Time taken to filter : {at1.duration_human}")

        # raise modal warning if no results due to too restrictive filtering
        if len(df_fit_results_filtered) == 0:
            return dash.no_update, True

        return df_fit_results_filtered.to_dict("records"), dash.no_update

    #%%

    # def list_is_none_or_empty(l):
    #     return l is None or len(l) == 0

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
        Output("dynamic_slider-container", "children"),
        Input("dropdown_slider", "value"),
        State("dynamic_slider-container", "children"),
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
        Output("main_graph", "figure"),
        Input("store", "data"),
        Input("tabs", "active_tab"),
    )
    def update_main_graph(data, active_tab):
        if active_tab is None:
            print("update_main_graph got active_tab == None")
        if data is None:
            print("update_main_graph got data == None")

        figure = get_main_figure(data, active_tab)

        # allows to size of marker to change without loosing zoom level
        if active_tab == "fig_fit_results":
            figure["layout"]["uirevision"] = True

        return figure

    @app.callback(
        Output("main_graph_div", "children"),
        Input("tabs", "active_tab"),
        Input({"type": "button_reset", "index": ALL}, "n_clicks"),
        State("store", "data"),
    )
    def update_tab_layout(active_tab, button_n, data):
        if active_tab is None:
            print("update_tab_layout got active_tab == None")
        if data is None:
            # print("update_tab_layout got data == None")
            raise PreventUpdate

        return make_tab_from_data(data, active_tab)

    #%%

    @app.callback(
        Output("graph_mismatch", "figure"),
        Input("tabs", "active_tab"),
        Input("dropdown_mismatch_tax_id", "value"),
        Input("dropdown_mismatch_shortname", "value"),
    )
    def update_mismatch_plot(active_tab, dropdown_mismatch_tax_id, dropdown_name):

        if dropdown_mismatch_tax_id is None:
            if active_tab == "fig_histograms":
                s = "Does not work for binned data"
                return dashboard.figures.create_empty_figure(s=s)
            else:
                return dashboard.figures.create_empty_figure()

        try:
            group = fit_results.get_single_count_group(
                shortname=dropdown_name,
                tax_id=dropdown_mismatch_tax_id,
            )
            fit = fit_results.get_single_fit_prediction(
                shortname=dropdown_name,
                tax_id=dropdown_mismatch_tax_id,
            )
            chosen_mismatch_columns = ["C→T", "G→A"]
            fig = dashboard.figures.plot_count_fraction(
                group,
                chosen_mismatch_columns,
                fit,
            )
            return fig

        # when selecting histogram without customdata
        except KeyError:
            raise PreventUpdate

    #%%

    @app.callback(
        Output("data_table", "data"),
        Input("main_graph", "clickData"),
        Input("tabs", "active_tab"),
    )
    def update_data_table(click_data, active_tab):
        if click_data is None:
            if active_tab == "fig_histograms":
                s = "Does not work for binned data (histograms)"
                ds = dashboard.datatable.create_empty_dataframe_for_datatable(s)
            else:
                ds = dashboard.datatable.create_empty_dataframe_for_datatable()
            return ds.to_dict("records")

        try:
            tax_id = fit_results.parse_click_data(click_data, column="tax_id")
            shortname = fit_results.parse_click_data(click_data, column="shortname")
            df_fit_results_filtered = fit_results.filter(
                {"shortname": shortname, "tax_id": tax_id}
            )
            return df_fit_results_filtered.to_dict("records")

        # when selecting histogram without customdata
        except KeyError:
            raise PreventUpdate

    #%%

    def get_tax_id_options_based_on_shortname(shortname, df_string="df_fit_results"):
        """df_string is a string, eg. df_fit_results or  df_counts.
        The 'df_' part is optional
        """
        tax_ids = sorted(
            fit_results.load_df_counts_shortname(shortname, columns="tax_id")[
                "tax_id"
            ].unique()
        )
        options = [{"label": i, "value": i} for i in tax_ids]
        return options

    @app.callback(
        Output("dropdown_mismatch_tax_id", "options"),
        Input("dropdown_mismatch_shortname", "value"),
    )
    def update_dropdown_tax_id_options(shortname):
        # if shortname is None:
        # print("update_dropdown_tax_id_options got shortname==None")
        return get_tax_id_options_based_on_shortname(shortname, df_string="counts")

    @app.callback(
        Output("dropdown_mismatch_shortname", "value"),
        Output("dropdown_mismatch_tax_id", "value"),
        Input("main_graph", "clickData"),
        State("tabs", "active_tab"),
    )
    def update_dropdowns_based_on_click_data(click_data, active_tab):
        if click_data is not None:
            if active_tab == "fig_histograms":
                # print("update_dropdowns_based_on_click_data got here")
                raise PreventUpdate
            try:
                tax_id = fit_results.parse_click_data(click_data, column="tax_id")
                shortname = fit_results.parse_click_data(click_data, column="shortname")
                return shortname, tax_id
            except KeyError:
                # print("update_dropdowns_based_on_click_data got KeyError")
                raise PreventUpdate
                # return None, None
        else:
            # print("update_dropdowns_based_on_click_data got click_data == None")
            raise PreventUpdate

    #%%

    def include_subspecies(subspecies):
        if len(subspecies) == 1:
            return True
        return False

    @app.callback(
        Output("tax_id_filter_counts_output", "children"),
        # Input("tax_id_filter_button", "n_clicks"),
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

    return app


#%%


# def is_ipython():
#     return hasattr(__builtins__, "__IPYTHON__")


# if __name__ == "__main__" and not is_ipython():
#     app.run_server(debug=True)
#     # app.run_server(debug=False)


# else:

#     dashboard.elements.get_range_slider_keywords(
#         fit_results,
#         column="D_max",
#         N_steps=100,
#     )
#     dashboard.elements.get_range_slider_keywords(
#         fit_results,
#         column="n_sigma",
#         N_steps=100,
#     )

#     shortname = "KapK-12-1-24-Ext-1-Lib-1-Index2"
#     # shortname = "SJArg-1-Nit"
#     tax_id = 131567

#     df_fit_results_all = fit_results.df_fit_results

#     df_fit_results = fit_results.filter({"tax_id": tax_id, "shortname": shortname})

#     group = fit_results.get_single_count_group(shortname=shortname, tax_id=tax_id)
#     fit = fit_results.get_single_fit_prediction(shortname=shortname, tax_id=tax_id)

#     chosen_mismatch_columns = ["C→T", "G→A"]

#     # # #%%

#     fig = dashboard.figures.plot_count_fraction(group, chosen_mismatch_columns, fit=fit)
#     fig

#     group[["position", "CT", "C"]]

#     reload(taxonomy)
#     tax_name = "Ursus"
#     tax_name = "Mammalia"
#     tax_ids = taxonomy.extract_descendant_tax_ids(tax_name)

#     df_fit_results = fit_results.filter({"tax_ids": tax_ids})

#     df = df_fit_results_all.query("k_sum_total > 1_000_000")
#     N_shortnames = df.shortname.nunique()
#     symbol_sequence = [str(i) for i in range(N_shortnames)]

#     fig1 = dashboard.figures.plot_fit_results(fit_results, df)
#     # fig2 = dashboard.figures.plot_histograms(fit_results, df)
#     # fig3 = dashboard.figures.plot_scatter_matrix(fit_results, df)
#     # fig4 = dashboard.figures.plot_forward_reverse(fit_results, df)
