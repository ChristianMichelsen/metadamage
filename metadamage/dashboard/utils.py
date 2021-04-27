# Scientific Library
import numpy as np

# Standard Library
from collections import namedtuple
from threading import Timer
import webbrowser

# Third Party
from PIL import ImageColor
from dash.exceptions import PreventUpdate

#%%
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.io as pio

# First Party
from metadamage import taxonomy
from metadamage.utils import human_format


def set_custom_theme():

    pio.templates["custom_template"] = go.layout.Template(
        layout=go.Layout(
            font_size=16,
            title_font_size=30,
            legend=dict(
                title_font_size=20,
                font_size=16,
                itemsizing="constant",
                itemclick=False,
                itemdoubleclick=False,
            ),
            hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
            dragmode="zoom",
        )
    )

    # pio.templates.default = "plotly_white"
    pio.templates.default = "simple_white+custom_template"

    return None


#%%


def is_log_transform_column(column):
    log_transform_columns = ["N_alignments", "k_sum_", "N_sum_"]
    return any([log_col in column for log_col in log_transform_columns])


def log_transform_slider(x):
    return np.where(x < 0, 0, 10 ** np.clip(x, 0, a_max=None))


def open_browser():
    # webbrowser.open_new("http://localhost:8050")
    webbrowser.open("http://localhost:8050")


def open_browser_in_background():
    Timer(3, open_browser).start()


#%%


def hex_to_rgb(hex_string, opacity=1):
    rgb = ImageColor.getcolor(hex_string, "RGB")
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


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

    if is_log_transform_column(column):
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

        f = lambda x: human_format(log_transform_slider(x))
        marks = {int(i): f"{f(i)}" for i in marks_steps}

        marks[marks_steps[0]] = {"label": no_min, "style": {"color": "#a3ada9"}}
        marks[marks_steps[-1]] = {"label": no_max, "style": {"color": "#a3ada9"}}

    elif column in ["D_max", "q", "A", "c"]:
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


def get_shortname_tax_id_from_click_data(fit_results, click_data):
    try:
        shortname = fit_results.parse_click_data(click_data, column="shortname")
        tax_id = fit_results.parse_click_data(click_data, column="tax_id")
    except KeyError:
        raise PreventUpdate
    return shortname, tax_id


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


def apply_tax_id_filter(fit_results, d_filter, tax_id_filter_input):
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


#%%


def key_is_in_list_case_insensitive(lst, key):
    return any([key.lower() in s.lower() for s in lst])


#%%


def get_configurations(
    sidebar_filter_width=30,  # in %
    sidebar_plot_width=20,  # in %
    content_main_margin=1,  # in %
):

    style_sidebar_base = {
        "position": "fixed",
        "top": 62.5,
        "bottom": 0,
        "height": "100%",
        "z-index": 1,
        "overflow-x": "hidden",
        "transition": "all 0.5s",
        "padding": "0.5rem 1rem",
        # "background-color": "#f8f9fa",
    }

    # the style arguments for the sidebar_plot. We use position:fixed and a fixed width
    style_sidebar_filter_shown = {
        **style_sidebar_base,
        "left": "0%",
        "width": f"{sidebar_filter_width}%",
    }

    style_sidebar_filter_hidden = {
        **style_sidebar_base,
        "left": f"-{sidebar_filter_width}%",
        "width": f"{sidebar_filter_width}%",
    }

    # the style arguments for the sidebar_plot. We use position:fixed and a fixed width
    style_sidebar_plot_shown = {
        **style_sidebar_base,
        "left": f"{100-sidebar_plot_width}%",
        "width": f"{sidebar_plot_width}%",
    }

    style_sidebar_plot_hidden = {
        **style_sidebar_base,
        "left": "100%",
        "width": f"{sidebar_plot_width}%",
    }

    style_main_base = {
        "transition": "margin .5s",
        "padding": "2rem 1rem",
        # "background-color": "#f8f9fa",
    }

    style_main_both_sidebars = {
        **style_main_base,
        "margin-left": f"{sidebar_filter_width+content_main_margin}%",
        "margin-right": f"{sidebar_plot_width+content_main_margin}%",
    }

    style_main_no_sidebars = {
        **style_main_base,
        "margin-left": f"{content_main_margin}%",
        "margin-right": f"{content_main_margin}%",
    }

    style_main_filter_sidebar = {
        **style_main_base,
        "margin-left": f"{sidebar_filter_width+content_main_margin}%",
        "margin-right": f"{content_main_margin}%",
    }

    style_main_plot_sidebar = {
        **style_main_base,
        "margin-left": f"{content_main_margin}%",
        "margin-right": f"{sidebar_plot_width+content_main_margin}%",
    }

    configuration = namedtuple(
        "configuration",
        [
            "style_content_main",
            "style_sidebar_filter",
            "style_sidebar_plot",
            "state_sidebar_filter",
            "state_sidebar_plot",
        ],
    )

    d_sidebar_filter = {
        "shown": {
            "style_sidebar_filter": style_sidebar_filter_shown,
            "state_sidebar_filter": "SHOWN",
        },
        "hidden": {
            "style_sidebar_filter": style_sidebar_filter_hidden,
            "state_sidebar_filter": "HIDDEN",
        },
    }

    d_sidebar_plot = {
        "shown": {
            "style_sidebar_plot": style_sidebar_plot_shown,
            "state_sidebar_plot": "SHOWN",
        },
        "hidden": {
            "style_sidebar_plot": style_sidebar_plot_hidden,
            "state_sidebar_plot": "HIDDEN",
        },
    }

    configurations = {
        1: configuration(
            style_content_main=style_main_no_sidebars,
            **d_sidebar_filter["hidden"],
            **d_sidebar_plot["hidden"],
        ),
        2: configuration(
            style_content_main=style_main_filter_sidebar,
            **d_sidebar_filter["shown"],
            **d_sidebar_plot["hidden"],
        ),
        3: configuration(
            style_content_main=style_main_plot_sidebar,
            **d_sidebar_filter["hidden"],
            **d_sidebar_plot["shown"],
        ),
        4: configuration(
            style_content_main=style_main_both_sidebars,
            **d_sidebar_filter["shown"],
            **d_sidebar_plot["shown"],
        ),
    }

    return configurations


#%%


def toggle_plot(
    configurations,
    current_state_sidebar_filter,
    current_state_sidebar_plot,
):

    # going from (4) -> (2)
    if (
        current_state_sidebar_filter == "SHOWN"
        and current_state_sidebar_plot == "SHOWN"
    ):
        return configurations[2]

    # going from (2) -> (4)
    elif (
        current_state_sidebar_filter == "SHOWN"
        and current_state_sidebar_plot == "HIDDEN"
    ):
        return configurations[4]

    # going from (3) -> (1)
    elif (
        current_state_sidebar_filter == "HIDDEN"
        and current_state_sidebar_plot == "SHOWN"
    ):
        return configurations[1]

    # going from (1) -> (3)
    elif (
        current_state_sidebar_filter == "HIDDEN"
        and current_state_sidebar_plot == "HIDDEN"
    ):
        return configurations[3]


#%%


def toggle_filter(
    configurations,
    current_state_sidebar_filter,
    current_state_sidebar_plot,
):

    # going from (4) -> (3)
    if (
        current_state_sidebar_filter == "SHOWN"
        and current_state_sidebar_plot == "SHOWN"
    ):
        return configurations[3]

    # going from (2) -> (1)
    elif (
        current_state_sidebar_filter == "SHOWN"
        and current_state_sidebar_plot == "HIDDEN"
    ):
        return configurations[1]

    # going from (3) -> (4)
    elif (
        current_state_sidebar_filter == "HIDDEN"
        and current_state_sidebar_plot == "SHOWN"
    ):
        return configurations[4]

    # going from (1) -> (2)
    elif (
        current_state_sidebar_filter == "HIDDEN"
        and current_state_sidebar_plot == "HIDDEN"
    ):
        return configurations[2]


def get_button_id(ctx):
    " Get button clicked"
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return button_id


def get_graph_kwargs():
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
    return graph_kwargs


def get_graph_kwargs_no_buttons():
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
    return graph_kwargs_no_buttons


def get_d_columns_latex():
    d_columns_latex = {
        "LR": r"$\lambda_\text{LR}$",
        "D_max": r"$D_\text{max}$",
        "q": r"$q$",
        "phi": r"$\phi$",
        "A": r"$A$",
        "c": r"$c$",
        "asymmetry": r"$\text{asymmetry}$",
        "D_max_significance": r"$Z_{D_\text{max}}$",
        "rho_Ac": r"$\rho_{A, c}$",
        "rho_Ac_abs": r"$|\rho_{A, c}|$",
        "LR_P": r"$\text{P}_\lambda$",
        "LR_n_sigma": r"$\sigma_\lambda$",
        #
        "D_max_std": r"$\sigma_{D_\text{max}}$",
        "q_std": r"$\sigma_q$",
        "phi_std": r"$\sigma_\phi$",
        "A_std": r"$\sigma_A$",
        "c_std": r"$\sigma_c$",
        #
        "N_alignments": r"$N_\text{alignments}$",
        "k_sum_total": r"$\sum_i k_i$",
        "N_sum_total": r"$\sum_i N_i$",
        "N_min": r"$\text{min} N_i$",
        #
        "log_LR": r"$\log_{10}(1+\lambda_\text{LR})$",
        "log_phi": r"$\log_{10}(1+\phi)$",
        "log_N_alignments": r"$\log_{10}(1+N_\text{alignments})$",
        "log_k_sum_total": r"$\log_{10}(1+\sum_i k_i)$",
        "log_N_sum_total": r"$\log_{10}(1+\sum_i N_i)$",
        #
        "forward_LR": r"$ \lambda_\text{LR} \,\, \text{(forward)}$",
        "forward_D_max": r"$ D\text{max} \,\, \text{(forward)}$",
        "forward_q": r"$ q \,\, \text{(forward)}$",
        "forward_phi": r"$ \phi \,\, \text{(forward)}$",
        "forward_A": r"$ A \,\, \text{(forward)}$",
        "forward_c": r"$ c \,\, \text{(forward)}$",
        "forward_rho_Ac": r"$ \rho_{A, c} \,\, \text{(forward)}$",
        "forward_LR_P": r"$ \text{P}_\lambda \,\, \text{(forward)}$",
        "forward_LR_n_sigma": r"$ \sigma_\lambda \,\, \text{(forward)}$",
        #
        "forward_D_max_std": r"$ \sigma_{D_\text{max}} \,\, \text{(forward)}$",
        "forward_q_std": r"$ \sigma_q \,\, \text{(forward)}$",
        "forward_phi_std": r"$ \sigma_\phi \,\, \text{(forward)}$",
        "forward_A_std": r"$ \sigma_A \,\, \text{(forward)}$",
        "forward_c_std": r"$ \sigma_c \,\, \text{(forward)}$",
        #
        "k_sum_forward": r"$\sum_i k_i \,\, \text{(forward)}$",
        "N_z1_forward": r"$N_{z=1} \,\, \text{(forward)}$",
        "N_sum_forward": r"$\sum_i N_i \,\, \text{(forward)}$",
        #
        "reverse_LR": r"$ \lambda_\text{LR} \,\, \text{(reverse)}$",
        "reverse_D_max": r"$ D\text{max} \,\, \text{(reverse)}$",
        "reverse_q": r"$ q \,\, \text{(reverse)}$",
        "reverse_phi": r"$ \phi \,\, \text{(reverse)}$",
        "reverse_A": r"$ A \,\, \text{(reverse)}$",
        "reverse_c": r"$ c \,\, \text{(reverse)}$",
        "reverse_rho_Ac": r"$ \rho_{A, c} \,\, \text{(reverse)}$",
        "reverse_LR_P": r"$ \text{P}_\lambda \,\, \text{(reverse)}$",
        "reverse_LR_n_sigma": r"$ \sigma_\lambda \,\, \text{(reverse)}$",
        #
        "reverse_D_max_std": r"$ \sigma_{D_\text{max}} \,\, \text{(reverse)}$",
        "reverse_q_std": r"$ \sigma_q \,\, \text{(reverse)}$",
        "reverse_phi_std": r"$ \sigma_\phi \,\, \text{(reverse)}$",
        "reverse_A_std": r"$ \sigma_A \,\, \text{(reverse)}$",
        "reverse_c_std": r"$ \sigma_c \,\, \text{(reverse)}$",
        #
        "k_sum_reverse": r"$\sum_i k_i \,\, \text{(reverse)}$",
        "N_z1_reverse": r"$N_{z=1} \,\, \text{(reverse)}$",
        "N_sum_reverse": r"$\sum_i N_i \,\, \text{(reverse)}$",
        #
        "Bayesian_n_sigma": r"$n_\sigma \,\, \text{(Bayesian)}$",
        "Bayesian_D_max": r"$D_\text{max} \,\, \text{(Bayesian)}$",
        "Bayesian_q": r"$q \,\, \text{(Bayesian)}$",
        "Bayesian_phi": r"$\phi \,\, \text{(Bayesian)}$",
        "Bayesian_A": r"$A \,\, \text{(Bayesian)}$",
        "Bayesian_c": r"$c \,\, \text{(Bayesian)}$",
        "Bayesian_D_max_std": r"$\sigma_{D_\text{max}} \,\, \text{(Bayesian)}$",
        #
    }

    columns = list(d_columns_latex.keys())
    columns_no_log = [col for col in columns if not col.startswith("log_")]

    return d_columns_latex, columns, columns_no_log
