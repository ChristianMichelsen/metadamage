# Scientific Library
import numpy as np
import pandas as pd

# Third Party
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# First Party
from metadamage import counts, dashboard


#%%


def create_empty_figure(s=None, width=None, height=None):

    if s is None:
        s = "Please select a point"

    fig = go.Figure()

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        text=s,
        font_size=20,
        showarrow=False,
    )

    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        width=width,
        height=height,
    )

    if width is not None:
        fig.update_layout(width=width)

    if height is not None:
        fig.update_layout(height=height)

    return fig


#%%


def set_opacity_for_trace(
    trace, method="sqrt", scale=3.0, opacity_min=0.001, opacity_max=0.9
):
    N = len(trace.x)
    if "lin" in method:
        opacity = 1 / N
    elif method == "sqrt":
        opacity = 1 / np.sqrt(N)
    elif method == "log":
        opacity = 1 / np.log(N)

    opacity *= scale
    opacity = max(opacity_min, min(opacity, opacity_max))

    # print(trace.name, opacity)
    trace.update(marker_opacity=opacity)
