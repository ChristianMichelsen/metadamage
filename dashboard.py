import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px


import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


df = pd.read_csv("https://plotly.github.io/datasets/country_indicators.csv")
available_indicators = df["Indicator Name"].unique()


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

# (1) No sidebars, (2) Only left filter sidebar,
# (3) Only right filter sidebar, (4) Both sidebars
start_configuration_id = 1

sidebar_filter_width = 30  # in %
sidebar_plot_width = 20  # in %
content_main_margin = 1  # in %

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


from collections import namedtuple

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


start_configuration = configurations[start_configuration_id]

dropdown_x_axis = dcc.Dropdown(
    id="xaxis-column",
    options=[{"label": i, "value": i} for i in available_indicators],
    value="Fertility rate, total (births per woman)",
)

lin_log_scale_x_axis = dcc.RadioItems(
    id="xaxis-type",
    options=[{"label": i, "value": i} for i in ["Linear", "Log"]],
    value="Linear",
    labelStyle={"display": "inline-block"},
)

dropdown_y_axis = dcc.Dropdown(
    id="yaxis-column",
    options=[{"label": i, "value": i} for i in available_indicators],
    value="Life expectancy at birth, total (years)",
)

lin_log_scale_y_axis = dcc.RadioItems(
    id="yaxis-type",
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


slider = dcc.Slider(
    id="year--slider",
    min=df["Year"].min(),
    max=df["Year"].max(),
    value=df["Year"].max(),
    marks={str(year): str(year) for year in df["Year"].unique()},
    step=None,
)

content_main = html.Div(
    html.Div(
        [
            html.Div([div_x_axis, div_y_axis]),
            dcc.Graph(id="indicator-graphic"),
            # slider,
        ]
    ),
    id="content_main",
    style=start_configuration.style_content_main,
)


sidebar_filter = html.Div(
    [
        html.H2("Filter", className="display-4"),
        html.Hr(),
        html.P("filter here", className="lead"),
    ],
    id="sidebar_filter",
    style=start_configuration.style_sidebar_filter,
)


sidebar_plot = html.Div(
    [
        html.H2("Plot", className="display-4"),
        html.Hr(),
        html.P("asdasdasd", className="lead"),
    ],
    id="sidebar_plot",
    style=start_configuration.style_sidebar_plot,
)

app.layout = html.Div(
    [
        dcc.Store(id="sidebar_plot_state"),
        dcc.Store(id="sidebar_filter_state"),
        navbar,
        sidebar_filter,
        content_main,
        sidebar_plot,
    ],
)


@app.callback(
    Output("indicator-graphic", "figure"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("xaxis-type", "value"),
    Input("yaxis-type", "value"),
    # Input("year--slider", "value"),
)
def update_graph(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    # year_value,
):
    # dff = df[df["Year"] == year_value]

    dff = df

    fig = px.scatter(
        x=dff[dff["Indicator Name"] == xaxis_column_name]["Value"],
        y=dff[dff["Indicator Name"] == yaxis_column_name]["Value"],
        hover_name=dff[dff["Indicator Name"] == yaxis_column_name]["Country Name"],
    )

    fig.update_layout(margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest")

    fig.update_xaxes(
        title=xaxis_column_name, type="linear" if xaxis_type == "Linear" else "log"
    )

    fig.update_yaxes(
        title=yaxis_column_name, type="linear" if yaxis_type == "Linear" else "log"
    )

    return fig


#%%


def get_button_id(ctx):
    " Get button clicked"
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return button_id


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

    button_id = get_button_id(dash.callback_context)

    # if the toggle filter button was clicked
    if button_id == "btn_toggle_filter":
        return toggle_filter(current_state_sidebar_filter, current_state_sidebar_plot)

    # if the toggle plot button was clicked
    elif button_id == "btn_toggle_plot":
        return toggle_plot(current_state_sidebar_filter, current_state_sidebar_plot)

    # base configuration
    else:
        return start_configuration


def toggle_plot(current_state_sidebar_filter, current_state_sidebar_plot):

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


def toggle_filter(current_state_sidebar_filter, current_state_sidebar_plot):

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


if __name__ == "__main__":
    app.run_server(debug=True)
