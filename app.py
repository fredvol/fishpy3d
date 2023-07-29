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


# %% create input
def create_input(i, fishkite):
    input_list = [
        html.H6(f"{fishkite.name}"),
        html.Label("rising Angle (deg)"),
        dcc.Slider(
            id=f"slider-rising_angle_{i}",
            min=1,
            max=90,
            step=1,
            value=25,
            marks={i: str(i) for i in range(0, 90, 5)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Area"),
        dcc.Slider(
            id=f"slider-area_{i}",
            min=1,
            max=40,
            step=1,
            value=20,
            marks={i: str(i) for i in range(0, 40, 5)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Cl"),
        dcc.Slider(
            id=f"slider-cl_{i}",
            min=0,
            max=1,
            step=0.1,
            value=0.4,
            marks={i: str(i) for i in range(0, 1, 1)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("efficiency_angle"),
        dcc.Slider(
            id=f"slider-efficiency_angle_{i}",
            min=0,
            max=90,
            step=1,
            value=18,
            marks={i: str(i) for i in range(0, 90, 5)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Br(),
    ]

    input_div = html.Div(
        children=input_list,
        style={"padding": 10, "flex": 1},
    )
    return input_div


# %%"
app = dash.Dash(__name__)


fig = proj.plot()

list_of_controls = []

output_items = html.Div(
    children=[
        html.Label("Outputs"),
        html.Div(id="my-output"),
        dcc.Graph(id="fig1", figure=fig),
    ],
    style={"padding": 10, "flex": 1},
)

for i, fk in enumerate(proj.lst_fishkite):
    list_of_controls.append(create_input(i, fk))

list_of_controls.append(output_items)

app.layout = html.Div(
    list_of_controls,
    style={"display": "flex", "flex-direction": "column"},
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
        Input(component_id="slider-rising_angle_0", component_property="value"),
        Input(component_id="slider-area_0", component_property="value"),
        Input(component_id="slider-cl_0", component_property="value"),
        Input(component_id="slider-efficiency_angle_0", component_property="value"),
        Input(component_id="slider-rising_angle_1", component_property="value"),
        Input(component_id="slider-area_1", component_property="value"),
        Input(component_id="slider-cl_1", component_property="value"),
        Input(component_id="slider-efficiency_angle_1", component_property="value"),
    ],
)
def update(risingA0, area0, cl0, ef0, risingA1, area1, cl1, ef1):
    proj.lst_fishkite[0].rising_angle = risingA0
    proj.lst_fishkite[0].kite.area = area0
    proj.lst_fishkite[0].kite.cl = cl0
    proj.lst_fishkite[0].kite.efficiency_angle = ef0

    proj.lst_fishkite[1].rising_angle = risingA1
    proj.lst_fishkite[1].kite.area = area1
    proj.lst_fishkite[1].kite.cl = cl1
    proj.lst_fishkite[1].kite.efficiency_angle = ef1
    fig = proj.plot()

    text_detail = f"Compute for : {proj.detail()}  "

    return fig, text_detail


if __name__ == "__main__":
    app.run(debug=True)

# %%
