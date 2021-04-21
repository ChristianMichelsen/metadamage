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
        "background-color": "#f8f9fa",
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
