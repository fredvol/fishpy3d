# plotly app

# %% import

from model import Deflector, FishKite, Project

import dash
from dash import dcc  # import dash_core_components as dcc   # from dash import dcc
from dash import html  # import dash_html_components as html  # from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import copy
import pandas as pd

# %% Initial set up


wind_speed_i = 15  # kt
rising_angle_1 = 33  # deg
rising_angle_2 = 20  # deg

d_kite1 = Deflector("kite1", cl=0.4, cl_range=(0.4, 0.9), area=24, efficiency_angle=12)
d_fish1 = Deflector("fish1", cl=0.2, cl_range=(0.2, 0.4), area=0.1, efficiency_angle=14)

d_kite2 = Deflector("kite2", cl=0.6, cl_range=(0.4, 0.9), area=25, efficiency_angle=4)
d_fish2 = Deflector("fish2", cl=0.4, cl_range=(0.2, 0.4), area=0.07, efficiency_angle=8)

fk1 = FishKite("fk1", wind_speed_i, rising_angle_1, fish=d_fish1, kite=d_kite1)
fk2 = FishKite("fk2", wind_speed_i, rising_angle_2, fish=d_fish2, kite=d_kite2)

proj = Project([fk1, fk2])

# %%"
app = dash.Dash(__name__)


fig = proj.plot()


app.layout = html.Div(
    [
        html.H2("FishPy"),
        html.Div(
            [
                "Inputs: ",
                html.H6("Area"),
                dcc.Slider(
                    id="slider-AREA",
                    min=1,
                    max=50,
                    step=1,
                    value=20,
                    marks={i: str(i) for i in range(0, 50, 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.H6("Cl"),
                dcc.Slider(
                    id="sliderAR",
                    min=0.2,
                    max=1,
                    step=0.1,
                    value=0.7,
                    marks={i: str(i) for i in range(0, 20, 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dcc.Graph(id="fig1", figure=fig),
            ]
        ),
        html.Br(),
        html.Div(id="my-output"),
    ]
)


## callback
# @app.callback(
#     Output(component_id="my-output", component_property="children"),
#     [
#         Input(component_id="sliderAR", component_property="value"),
#         Input(component_id="slider-AREA", component_property="value"),
#     ],
# )
# def update_output_div(AR, area):
#     proj.lst_fishkite[1].kite.area = area
#     proj.lst_fishkite[1].kite.cl = AR

#     return f"Compute for : {proj.detail()}  "


@app.callback(
    Output("fig1", "figure"),
    Output(component_id="my-output", component_property="children"),
    [
        Input(component_id="sliderAR", component_property="value"),
        Input(component_id="slider-AREA", component_property="value"),
    ],
)
def update(AR, area):
    proj.lst_fishkite[1].kite.area = area
    proj.lst_fishkite[1].kite.cl = AR
    fig = proj.plot()

    text_detail = f"Compute for : {proj.detail()}  "

    return fig, text_detail


if __name__ == "__main__":
    app.run(debug=True)

# %%
