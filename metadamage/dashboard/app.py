# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from pathlib import Path

# Third Party
import dash
from dash.dependencies import ALL, Input, MATCH, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions.snippets import send_data_frame
import dash_html_components as html


def get_app(results_dir=Path("./data/out/results")):

    # First Party
    import metadamage as meta
    from metadamage import dashboard

    fit_results = dashboard.results.load(results_dir)

    #%%

    dashboard.utils.set_custom_theme()

    #%%

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

    #%%

    # (1) No sidebars, (2) Only left filter sidebar,
    # (3) Only right plot sidebar, (4) Both sidebars
    start_configuration_id = 3

    configurations = dashboard.utils.get_configurations(
        sidebar_left_width=30,
        sidebar_right_width=35,
        content_main_margin=1,
    )
    start_configuration = configurations[start_configuration_id]

    d_columns_latex, columns, columns_no_log = dashboard.utils.get_d_columns_latex()

    #%%

    app.layout = dashboard.content.get_app_layout(fit_results, start_configuration)

    #%%

    @app.callback(
        Output("navbar_collapsed_toggle_styling", "is_open"),
        Output("navbar_btn_toggle_styling", "outline"),
        Input("navbar_btn_toggle_styling", "n_clicks"),
        State("navbar_collapsed_toggle_styling", "is_open"),
    )
    def toggle_styling(n, is_open):
        # after click
        if n:
            return not is_open, is_open
        # initial setup
        return is_open, True

    #%%

    @app.callback(
        Output("sidebar_right_collapsed_toggle_combined", "is_open"),
        Output("sidebar_right_btn_toggle_combined", "outline"),
        Input("sidebar_right_btn_toggle_combined", "n_clicks"),
        State("sidebar_right_collapsed_toggle_combined", "is_open"),
    )
    def toggle_sidebar_right_combined(n, is_open):
        if n:
            return not is_open, is_open
        return is_open, False

    @app.callback(
        Output("sidebar_right_collapsed_toggle_fit_results", "is_open"),
        Output("sidebar_right_btn_toggle_fit_results", "outline"),
        Input("sidebar_right_btn_toggle_fit_results", "n_clicks"),
        State("sidebar_right_collapsed_toggle_fit_results", "is_open"),
    )
    def toggle_sidebar_right_fit_results(n, is_open):
        if n:
            return not is_open, is_open
        return is_open, True

    @app.callback(
        Output("sidebar_right_collapsed_toggle_forward_reverse", "is_open"),
        Output("sidebar_right_btn_toggle_forward_reverse", "outline"),
        Input("sidebar_right_btn_toggle_forward_reverse", "n_clicks"),
        State("sidebar_right_collapsed_toggle_forward_reverse", "is_open"),
    )
    def toggle_sidebar_right_forward_reverse(n, is_open):
        if n:
            return not is_open, is_open
        return is_open, True

    #%%

    @app.callback(
        Output("sidebar_right_graph_combined", "figure"),
        Input("main_graph", "clickData"),
    )
    def update_sidebar_right_plot_combined(click_data):
        return dashboard.figures.update_raw_count_plots(
            fit_results,
            click_data,
            forward_reverse="",
        )

    @app.callback(
        Output("sidebar_right_graph_forward", "figure"),
        Input("main_graph", "clickData"),
    )
    def update_sidebar_right_plot_forward(click_data):
        return dashboard.figures.update_raw_count_plots(
            fit_results,
            click_data,
            forward_reverse="Forward",
        )

    @app.callback(
        Output("sidebar_right_graph_reverse", "figure"),
        Input("main_graph", "clickData"),
    )
    def update_sidebar_right_plot_reverse(click_data):
        return dashboard.figures.update_raw_count_plots(
            fit_results,
            click_data,
            forward_reverse="Reverse",
        )

    @app.callback(
        Output("sidebar_right_datatable_fit_results", "children"),
        Input("main_graph", "clickData"),
    )
    def update_sidebar_right_datatable_fit_results(click_data):
        if click_data:
            shortname, tax_id = dashboard.utils.get_shortname_tax_id_from_click_data(
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

                f"lambda_LR: {ds['lambda_LR']:.2f}",html.Br(),
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

    @app.callback(
        Output("main_graph", "figure"),
        Output("modal", "is_open"),
        Input("sidebar_left_dropdown_samples", "value"),
        Input("sidebar_left_tax_id_input", "value"),
        Input("tax_id_plot_button", "n_clicks"),
        Input({"type": "sidebar_left_fit_results_dynamic", "index": ALL}, "value"),
        Input("xaxis_column", "value"),
        Input("yaxis_column", "value"),
        Input("marker_transformation_variable", "value"),
        Input("marker_transformation_function", "value"),
        Input("marker_transformation_slider", "value"),
        Input("modal_close_button", "n_clicks"),
        State({"type": "sidebar_left_fit_results_dynamic", "index": ALL}, "id"),
        State("sidebar_left_tax_id_input_descendants", "value"),
        State("sidebar_left_tax_id_subspecies", "value"),
        State("modal", "is_open"),
    )
    def update_main_graph(
        sidebar_left_dropdown_samples,
        sidebar_left_tax_id_input,
        tax_id_plot_button,
        sidebar_left_fit_results_dynamic_value,
        xaxis_column_name,
        yaxis_column_name,
        marker_transformation_variable,
        marker_transformation_function,
        marker_transformation_slider,
        modal_close_button,
        sidebar_left_fit_results_dynamic_ids,
        sidebar_left_tax_id_input_descendants,
        sidebar_left_tax_id_subspecies,
        modal,
    ):

        # if modal is open and the "close" button is clicked, close down modal
        if modal_close_button and modal:
            return dash.no_update, False

        # if no files selected
        if not sidebar_left_dropdown_samples:
            raise PreventUpdate

        fit_results.set_marker_size(
            marker_transformation_variable,
            marker_transformation_function,
            marker_transformation_slider,
        )

        d_filter = {"shortnames": sidebar_left_dropdown_samples}

        columns_no_log = [id["index"] for id in sidebar_left_fit_results_dynamic_ids]
        for shortname, values in zip(
            columns_no_log, sidebar_left_fit_results_dynamic_value
        ):
            d_filter[shortname] = values

        dashboard.utils.apply_sidebar_left_tax_id(
            fit_results,
            d_filter,
            sidebar_left_tax_id_input,
        )

        dashboard.utils.apply_tax_id_descendants_filter(
            d_filter,
            sidebar_left_tax_id_input_descendants,
            sidebar_left_tax_id_subspecies,
        )

        df_fit_results_filtered = fit_results.filter(d_filter)

        # raise modal warning if no results due to too restrictive filtering
        if len(df_fit_results_filtered) == 0:
            return dash.no_update, True

        fig = dashboard.figures.make_figure(
            fit_results,
            df=df_fit_results_filtered,
            xaxis_column_name=xaxis_column_name,
            yaxis_column_name=yaxis_column_name,
            d_columns_latex=d_columns_latex,
        )

        return fig, dash.no_update

    #%%

    @app.callback(
        Output("sidebar_left_dropdown_samples", "value"),
        Input("sidebar_left_dropdown_samples", "value"),
    )
    def update_dropdown_samples_when_Select_all(sidebar_left_dropdown_samples):
        if dashboard.utils.key_is_in_list_case_insensitive(
            sidebar_left_dropdown_samples,
            "Select all",
        ):
            sidebar_left_dropdown_samples = fit_results.shortnames
        elif dashboard.utils.key_is_in_list_case_insensitive(
            sidebar_left_dropdown_samples,
            "Default selection",
        ):
            sidebar_left_dropdown_samples = dashboard.utils.get_shortnames_each(
                fit_results.shortnames
            )

        sidebar_left_dropdown_samples = list(sorted(sidebar_left_dropdown_samples))

        return sidebar_left_dropdown_samples

    #%%

    @app.callback(
        Output("sidebar_left_fit_results_container", "children"),
        Input("sidebar_left_fit_results", "value"),
        State("sidebar_left_fit_results_container", "children"),
        State({"type": "sidebar_left_fit_results_dynamic", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_sidebar_left_fit_result_sliders(dropdown_names, children, current_ids):

        id_type = "dbc"

        current_names = dashboard.content.get_current_names(current_ids)

        # add new slider
        if dashboard.content.slider_is_added(current_names, dropdown_names):
            column = dashboard.content.get_name_of_added_slider(
                current_names, dropdown_names
            )
            new_element = dashboard.content.make_new_slider(
                fit_results, column, id_type=id_type
            )
            children.append(new_element)

        # remove selected slider
        else:
            column = dashboard.content.get_name_of_removed_slider(
                current_names, dropdown_names
            )
            dashboard.content.remove_name_from_children(
                column, children, id_type=id_type
            )

        return children

    @app.callback(
        Output(
            {"type": "sidebar_left_fit_results_dynamic_name", "index": MATCH},
            "children",
        ),
        Input({"type": "sidebar_left_fit_results_dynamic", "index": MATCH}, "value"),
        State({"type": "sidebar_left_fit_results_dynamic", "index": MATCH}, "id"),
        prevent_initial_call=True,
    )
    def update_sidebar_left_fit_result_slider_names(
        dynamic_slider_values, sidebar_left_fit_results_dynamic_name
    ):
        column = sidebar_left_fit_results_dynamic_name["index"]
        name = dashboard.content.get_slider_name(column, dynamic_slider_values)
        return name

    #%%

    @app.callback(
        Output("sidebar_left_tax_id_counts_output", "children"),
        Input("sidebar_left_tax_id_input_descendants", "value"),
        Input("sidebar_left_tax_id_subspecies", "value"),
    )
    def update_sidebar_left_sidebar_left_tax_id_counts(tax_name, subspecies):

        if tax_name is None or tax_name == "":
            return f"No specific Tax IDs selected, defaults to ALL."
            # raise PreventUpdate

        tax_ids = meta.taxonomy.extract_descendant_tax_ids(
            tax_name,
            include_subspecies=dashboard.utils.include_subspecies(subspecies),
        )
        N_tax_ids = len(tax_ids)
        if N_tax_ids == 0:
            return f"Couldn't find any Tax IDs for {tax_name} in NCBI"
        return (
            f"Found {meta.utils.human_format(N_tax_ids)} Tax IDs for {tax_name} in NCBI"
        )

    #%%

    @app.callback(
        Output("sidebar_left_samples_collapsed", "is_open"),
        Output("sidebar_left_samples_btn", "outline"),
        Input("sidebar_left_samples_btn", "n_clicks"),
        State("sidebar_left_samples_collapsed", "is_open"),
    )
    def toggle_sidebar_left_samples(n, is_open):
        # after click
        if n:
            return not is_open, is_open
        # initial setup
        return is_open, True

    @app.callback(
        Output("sidebar_left_taxanomics_collapsed", "is_open"),
        Output("sidebar_left_taxanomics_btn", "outline"),
        Input("sidebar_left_taxanomics_btn", "n_clicks"),
        State("sidebar_left_taxanomics_collapsed", "is_open"),
    )
    def toggle_sidebar_left_taxanomics(n, is_open):
        if n:
            return not is_open, is_open
        return is_open, True

    @app.callback(
        Output("sidebar_left_fit_results_collapsed", "is_open"),
        Output("sidebar_left_fit_results_btn", "outline"),
        Input("sidebar_left_fit_results_btn", "n_clicks"),
        State("sidebar_left_fit_results_collapsed", "is_open"),
    )
    def toggle_sidebar_left_fit_results(n, is_open):
        if n:
            return not is_open, is_open
        return is_open, True

    #%%

    @app.callback(
        Output("content_main", "style"),
        Output("sidebar_left", "style"),
        Output("sidebar_right", "style"),
        Output("sidebar_left_state", "data"),
        Output("sidebar_right_state", "data"),
        Output("sidebar_left_toggle_btn", "outline"),
        Output("sidebar_right_toggle_btn", "outline"),
        Input("sidebar_left_toggle_btn", "n_clicks"),
        Input("sidebar_right_toggle_btn", "n_clicks"),
        State("sidebar_left_state", "data"),
        State("sidebar_right_state", "data"),
        State("sidebar_left_toggle_btn", "outline"),
        State("sidebar_right_toggle_btn", "outline"),
    )
    def toggle_sidebars(
        _sidebar_left_toggle_btn,
        _sidebar_right_toggle_btn,
        current_state_sidebar_left,
        current_state_sidebar_right,
        sidebar_left_toggle_btn_outline,
        sidebar_right_toggle_btn_outline,
    ):

        button_id = dashboard.utils.get_button_id(dash.callback_context)

        # if the toggle filter button was clicked
        if button_id == "sidebar_left_toggle_btn":
            return (
                *dashboard.utils.toggle_filter(
                    configurations,
                    current_state_sidebar_left,
                    current_state_sidebar_right,
                ),
                not sidebar_left_toggle_btn_outline,
                sidebar_right_toggle_btn_outline,
            )

        # if the toggle plot button was clicked
        elif button_id == "sidebar_right_toggle_btn":
            return (
                *dashboard.utils.toggle_plot(
                    configurations,
                    current_state_sidebar_left,
                    current_state_sidebar_right,
                ),
                sidebar_left_toggle_btn_outline,
                not sidebar_right_toggle_btn_outline,
            )

        # base configuration
        else:
            return *start_configuration, True, False

    #%%

    @app.callback(
        Output("export", "data"),
        Input("navbar_btn_export", "n_clicks"),
        State("sidebar_left_dropdown_samples", "value"),
        State("sidebar_left_tax_id_input", "value"),
        State({"type": "sidebar_left_fit_results_dynamic", "index": ALL}, "value"),
        State({"type": "sidebar_left_fit_results_dynamic", "index": ALL}, "id"),
        State("sidebar_left_tax_id_input_descendants", "value"),
        State("sidebar_left_tax_id_subspecies", "value"),
    )
    def export(
        navbar_btn_export,
        sidebar_left_dropdown_samples,
        sidebar_left_tax_id_input,
        sidebar_left_fit_results_dynamic_value,
        sidebar_left_fit_results_dynamic_ids,
        sidebar_left_tax_id_input_descendants,
        sidebar_left_tax_id_subspecies,
    ):

        if navbar_btn_export:

            d_filter = {"shortnames": sidebar_left_dropdown_samples}

            columns_no_log = [
                id["index"] for id in sidebar_left_fit_results_dynamic_ids
            ]
            for shortname, values in zip(
                columns_no_log, sidebar_left_fit_results_dynamic_value
            ):
                d_filter[shortname] = values

            dashboard.utils.apply_sidebar_left_tax_id(
                fit_results,
                d_filter,
                sidebar_left_tax_id_input,
            )

            dashboard.utils.apply_tax_id_descendants_filter(
                d_filter,
                sidebar_left_tax_id_input_descendants,
                sidebar_left_tax_id_subspecies,
            )

            df_fit_results_filtered = fit_results.filter(d_filter)

            return send_data_frame(
                df_fit_results_filtered.loc[:, :"LCA"].to_csv,
                "filtered_results.csv",
                index=False,
            )

    #%%

    return app


#%%

# if __name__ == "__main__":
#     app.run_server(debug=True)
