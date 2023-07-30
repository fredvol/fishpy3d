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


output_items = html.Div(
    children=[
        html.Label("Outputs"),
        html.Div(id="my-output"),
        dcc.Graph(id="fig1", figure=fig),
    ],
    style={"padding": 10, "flex": 1},
)

list_of_controls = []
for i, fk in enumerate(proj.lst_fishkite):
    list_of_controls.append(create_input(i, fk))


list_of_controls.append(output_items)

app.layout = html.Div(
    list_of_controls,
    style={"display": "flex", "flex-direction": "row"},
)


## callback

from dash import ctx


@app.callback(
    Output("fig1", "figure"),
    Output(component_id="my-output", component_property="children"),
    inputs={
        "all_inputs": {
            0: {
                "rising_angle": Input("slider-rising_angle_0", "value"),
                "area": Input("slider-area_0", "value"),
                "cl": Input("slider-cl_0", "value"),
                "efficiency_angle": Input("slider-efficiency_angle_0", "value"),
            },
            1: {
                "rising_angle": Input("slider-rising_angle_1", "value"),
                "area": Input("slider-area_1", "value"),
                "cl": Input("slider-cl_1", "value"),
                "efficiency_angle": Input("slider-efficiency_angle_1", "value"),
            },
        }
    },
)
def update(all_inputs):
    c = ctx.args_grouping.all_inputs

    proj.lst_fishkite[0].rising_angle = c[0]["rising_angle"]["value"]
    proj.lst_fishkite[0].kite.area = c[0]["area"]["value"]
    proj.lst_fishkite[0].kite.cl = c[0]["cl"]["value"]
    proj.lst_fishkite[0].kite.efficiency_angle = c[0]["efficiency_angle"]["value"]

    proj.lst_fishkite[1].rising_angle = c[1]["rising_angle"]["value"]
    proj.lst_fishkite[1].kite.area = c[1]["area"]["value"]
    proj.lst_fishkite[1].kite.cl = c[1]["cl"]["value"]
    proj.lst_fishkite[1].kite.efficiency_angle = c[1]["efficiency_angle"]["value"]

    fig = proj.plot(add_background_image=True)

    text_detail = f"Compute for : {proj.detail()}  "

    return fig, text_detail


if __name__ == "__main__":
    app.run(debug=True)

# %%


# app.layout = dbc.Container(
#     [
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         html.H2("FishPy"),
#                         html.H5("V1"),
#                     ],
#                     width=True,
#                 ),
#             ],
#             align="end",
#         ),
#         html.Hr(),
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         html.H4("Key Parameters"),
#                         html.Div(
#                             [
#                                 html.H5("FishKite 1"),
#                                 # list_of_controls
#                                 html.Label("rising Angle (deg)"),
#                                 dcc.Slider(
#                                     id=f"slider-rising_angle_{0}",
#                                     min=1,
#                                     max=90,
#                                     step=1,
#                                     value=25,
#                                     marks={i: str(i) for i in range(0, 90, 5)},
#                                     tooltip={
#                                         "placement": "bottom",
#                                         "always_visible": True,
#                                     },
#                                 ),
#                                 html.Label("Area"),
#                                 dcc.Slider(
#                                     id=f"slider-area_{0}",
#                                     min=1,
#                                     max=40,
#                                     step=1,
#                                     value=20,
#                                     marks={i: str(i) for i in range(0, 40, 5)},
#                                     tooltip={
#                                         "placement": "bottom",
#                                         "always_visible": True,
#                                     },
#                                 ),
#                                 html.Label("Cl"),
#                                 dcc.Slider(
#                                     id=f"slider-cl_{0}",
#                                     min=0,
#                                     max=1,
#                                     step=0.1,
#                                     value=0.4,
#                                     marks={i: str(i) for i in range(0, 1, 1)},
#                                     tooltip={
#                                         "placement": "bottom",
#                                         "always_visible": True,
#                                     },
#                                 ),
#                                 html.Label("efficiency_angle"),
#                                 dcc.Slider(
#                                     id=f"slider-efficiency_angle_{0}",
#                                     min=0,
#                                     max=90,
#                                     step=1,
#                                     value=18,
#                                     marks={i: str(i) for i in range(0, 90, 5)},
#                                     tooltip={
#                                         "placement": "bottom",
#                                         "always_visible": True,
#                                     },
#                                 ),
#                                 html.Br(),
#                             ]
#                         ),
#                         html.Hr(),
#                         html.Div(
#                             [
#                                 html.H5("Aerodynamic Performance"),
#                                 dbc.Spinner(
#                                     html.P(id="my-output"),
#                                     color="primary",
#                                 ),
#                             ]
#                         ),
#                     ],
#                     width=3,
#                 ),
#                 dbc.Col(
#                     [
#                         # html.Div(id='display')
#                         dbc.Spinner(dcc.Graph(id="fig1", figure=fig))
#                     ],
#                     width=True,
#                 ),
#             ]
#         ),
#         html.Hr(),
#         html.P(
#             " FishKite design tools",
#         ),
#     ],
#     fluid=True,
# )
