from collections import namedtuple


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
        "background-color": "#f8f9fa",
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
    return d_columns_latex