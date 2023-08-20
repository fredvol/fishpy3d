# plotly app

# %% import
#  http://127.0.0.1:8050/
# git push heroku main
# https://fishpy2d-9143325b137e.herokuapp.com/

# imports
from model_3d import (
    Deflector,
    FishKite,
    Pilot,
    plot_3d_cases,
    plot_3d_cases_risingangle,
)

import dash

dash.register_page(__name__)

from dash import dcc  # import dash_core_components as dcc   # from dash import dcc
from dash import html  # import dash_html_components as html  # from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import copy
import pandas as pd

from app_components_3d import *
from dash import ctx, dash_table, callback

__version__ = "1.0.3"

# %% Initial set up
d_pilot = Pilot(mass=80, pilot_drag=0.25)
d_kite1 = Deflector(
    "kite1",
    cl=0.8,
    cl_range=(0.4, 1),
    flat_area=20,
    flat_ratio=0.85,
    flat_aspect_ratio=6,
    profil_drag_coeff=0.013,
    parasite_drag_pct=0.03,  # 0.69,
)
d_fish1 = Deflector(
    "fish1",
    cl=0.6,
    cl_range=(0.2, 0.6),
    flat_area=0.1,
    flat_ratio=0.64,
    profil_drag_coeff=0.01,
    flat_aspect_ratio=8.5,
    parasite_drag_pct=0.06,
)

fk1 = FishKite(
    "fk1",
    wind_speed=15,
    rising_angle=20,
    fish=d_fish1,
    kite=d_kite1,
    pilot=d_pilot,
    extra_angle=20,
    cable_length_fish=30,
    cable_length_kite=12,
    cable_strength=500,
    cx_cable_water=1,
    cx_cable_air=1,
    tip_fish_depth=0.5,
)

# %%"
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server
dfG = fk1.create_df()

fig_rising_angle = plot_3d_cases_risingangle(dfG)
fig_all_pts = plot_3d_cases(dfG)
# df_table = proj.perf_table()


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Modify Operating Conditions",
                            color="danger",
                            id="operating_button_3d",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    operating_slider_components,
                                )
                            ),
                            id="operating_collapse_3d",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Modify FishKite_1 Parameters", id="shape1_button_3d"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    create_fk_sliders(0),
                                    # dcc.Markdown("bidon")
                                )
                            ),
                            id="shape1_collapse_3d",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Modify FishKite_2 Parameters",
                            color="success",
                            id="shape2_button_3d",
                        ),
                        html.Hr(),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    create_fk_sliders(1),
                                    # dcc.Markdown("bidon")
                                )
                            ),
                            id="shape2_collapse_3d",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Export data (txt format)",
                            color="info",
                            id="coordinates_button_3d",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    dbc.Spinner(
                                        html.P(id="my-output_3d"),
                                        color="primary",
                                    ),
                                )
                            ),
                            id="coordinates_collapse_3d",
                            is_open=False,
                        ),
                        html.Hr(),
                        # dcc.Markdown("##### Commands"),
                        # dbc.Button(
                        #     "Analyze",
                        #     id="analyze", color="primary", style={"margin": "5px"}),
                        html.Hr(),
                        dcc.Markdown("##### Numerical Results"),
                        # dash_table.DataTable(
                        #     df_table.to_dict("records"),
                        #     [{"name": i, "id": i} for i in df_table.columns],
                        #     id="perf_table_3d",
                        # ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="selected Rising angle",
                                    children=[
                                        dcc.Graph(
                                            id="fig1_3d_rising_angle",
                                            figure=fig_rising_angle,
                                            style={
                                                "position": "fixed",  # that imobilised the graph
                                            },
                                        ),
                                        dbc.CardBody(
                                            create_polar_rising_sliders(),
                                            # dcc.Markdown("bidon")
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Overview",
                                    children=[
                                        dcc.Graph(
                                            id="fig1_3d_all_pts",
                                            figure=fig_all_pts,
                                            style={
                                                "position": "fixed",  # that imobilised the graph
                                            },
                                        )
                                    ],
                                ),
                            ]
                        )
                    ],
                    width=9,
                    align="start",
                ),
            ]
        ),
        dcc.Markdown(
            """
        ......................................  

        **Hypothesis:**
         * Extra angle > 1 deg


        **Legend:**
         * OP = Operation point
         * fk = Fish-Kite
         * Fluid ratio = Apparent Water Speed / Apparent wind Speed
        """
        ),
    ],
    fluid=True,
)

## callback


### Callback to make shape parameters menu expand
@callback(
    Output("shape1_collapse_3d", "is_open"),
    [Input("shape1_button_3d", "n_clicks")],
    [State("shape1_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make shape parameters menu expand
@callback(
    Output("shape2_collapse_3d", "is_open"),
    [Input("shape2_button_3d", "n_clicks")],
    [State("shape2_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make operating parameters menu expand
@callback(
    Output("operating_collapse_3d", "is_open"),
    [Input("operating_button_3d", "n_clicks")],
    [State("operating_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make coordinates menu expand
@callback(
    Output("coordinates_collapse_3d", "is_open"),
    [Input("coordinates_button_3d", "n_clicks")],
    [State("coordinates_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to update polar selected  rising angle
@callback(
    Output("fig1_3d_rising_angle", "figure"),
    [
        Input("slider-rising_angle_polar", "value"),
        Input("slider-wind_speed", "value"),
        Input("data_color_polar_rising", "value"),
        Input("data_symbol_polar_rising", "value"),
    ],
)
def update_polar_rising_angle(rising_angle, target_wind, color_data, symbol_data):
    if symbol_data == "None":
        symbol_data = None

    return plot_3d_cases_risingangle(
        dfG,
        target_rising_angle=rising_angle,
        target_wind=target_wind,
        what=color_data,
        symbol=symbol_data,
    )


## main collaback
# @callback(
#     Output("fig1_3d", "figure"),
#     Output(component_id="my-output_3d", component_property="children"),
#     # Output(component_id="perf_table", component_property="data"),
#     inputs={
#         "all_inputs": {
#             "general": {
#                 "wind_speed": Input("slider-wind_speed", "value"),
#                 "bool_orthogrid": Input("bool_orthogrid", "on"),
#                 # "bool_backgrdimg": Input("bool_backgrdimg", "on"),
#                 "bool_isospeed": Input("bool_isospeed", "on"),
#                 "bool_isoeft": Input("bool_isoeft", "on"),
#                 "bool_isofluid": Input("bool_isofluid", "on"),
#                 "graph_size": Input("slider-graph_size", "value"),
#             },
#             0: {
#                 "bool_fk": Input("boolean_0", "on"),
#                 "rising_angle": Input("slider-rising_angle_0", "value"),
#                 "kite_area": Input("slider-kite_area_0", "value"),
#                 "kite_cl": Input("slider-kite_cl_0", "value"),
#                 "kite_efficiency_angle": Input(
#                     "slider-kite_efficiency_angle_0", "value"
#                 ),
#                 "fish_area": Input("slider-fish_area_0", "value"),
#                 "fish_cl": Input("slider-fish_cl_0", "value"),
#                 "fish_efficiency_angle": Input(
#                     "slider-fish_efficiency_angle_0", "value"
#                 ),
#             },
#             1: {
#                 "bool_fk": Input("boolean_1", "on"),
#                 "rising_angle": Input("slider-rising_angle_1", "value"),
#                 "kite_area": Input("slider-kite_area_1", "value"),
#                 "kite_cl": Input("slider-kite_cl_1", "value"),
#                 "kite_efficiency_angle": Input(
#                     "slider-kite_efficiency_angle_1", "value"
#                 ),
#                 "fish_area": Input("slider-fish_area_1", "value"),
#                 "fish_cl": Input("slider-fish_cl_1", "value"),
#                 "fish_efficiency_angle": Input(
#                     "slider-fish_efficiency_angle_1", "value"
#                 ),
#             },
#         }
#     },
# )
# def update(all_inputs):
#     c = ctx.args_grouping.all_inputs

#     bool_orthogrid = c["general"]["bool_orthogrid"]["value"]
#     # bool_backgrdimg = c["general"]["bool_backgrdimg"]["value"]
#     bool_isospeed = c["general"]["bool_isospeed"]["value"]
#     bool_isoeft = c["general"]["bool_isoeft"]["value"]
#     bool_isofluid = c["general"]["bool_isofluid"]["value"]

#     proj.lst_fishkite[0].wind_speed = c["general"]["wind_speed"]["value"]
#     proj.lst_fishkite[1].wind_speed = c["general"]["wind_speed"]["value"]

#     proj.lst_fishkite[0].rising_angle = c[0]["rising_angle"]["value"]
#     proj.lst_fishkite[0].kite.area = c[0]["kite_area"]["value"]
#     proj.lst_fishkite[0].kite.cl = c[0]["kite_cl"]["value"][1]
#     proj.lst_fishkite[0].kite.cl_range["min"] = c[0]["kite_cl"]["value"][0]
#     proj.lst_fishkite[0].kite.cl_range["max"] = c[0]["kite_cl"]["value"][2]
#     proj.lst_fishkite[0].kite.efficiency_angle = c[0]["kite_efficiency_angle"]["value"]
#     proj.lst_fishkite[0].fish.area = c[0]["fish_area"]["value"]
#     proj.lst_fishkite[0].fish.cl = c[0]["fish_cl"]["value"][1]
#     proj.lst_fishkite[0].fish.cl_range["min"] = c[0]["fish_cl"]["value"][0]
#     proj.lst_fishkite[0].fish.cl_range["max"] = c[0]["fish_cl"]["value"][2]
#     proj.lst_fishkite[0].fish.efficiency_angle = c[0]["fish_efficiency_angle"]["value"]

#     proj.lst_fishkite[1].rising_angle = c[1]["rising_angle"]["value"]
#     proj.lst_fishkite[1].kite.area = c[1]["kite_area"]["value"]
#     proj.lst_fishkite[1].kite.cl = c[1]["kite_cl"]["value"][1]
#     proj.lst_fishkite[1].kite.cl_range["min"] = c[1]["kite_cl"]["value"][0]
#     proj.lst_fishkite[1].kite.cl_range["max"] = c[1]["kite_cl"]["value"][2]
#     proj.lst_fishkite[1].kite.efficiency_angle = c[1]["kite_efficiency_angle"]["value"]
#     proj.lst_fishkite[1].fish.area = c[1]["fish_area"]["value"]
#     proj.lst_fishkite[1].fish.cl = c[1]["fish_cl"]["value"][1]
#     proj.lst_fishkite[1].fish.cl_range["min"] = c[1]["fish_cl"]["value"][0]
#     proj.lst_fishkite[1].fish.cl_range["max"] = c[1]["fish_cl"]["value"][2]
#     proj.lst_fishkite[1].fish.efficiency_angle = c[1]["fish_efficiency_angle"]["value"]

#     case_to_plot = []

#     if c[0]["bool_fk"]["value"]:
#         case_to_plot.append(proj.lst_fishkite[0])
#     if c[1]["bool_fk"]["value"]:
#         case_to_plot.append(proj.lst_fishkite[1])

#     fig = plot_cases(
#         list_of_cases=case_to_plot,
#         draw_ortho_grid=bool_orthogrid,
#         draw_iso_speed=bool_isospeed,
#         draw_iso_eft=bool_isoeft,
#         draw_iso_fluid=bool_isofluid,
#         add_background_image=False,
#         height_size=c["general"]["graph_size"]["value"],
#     )

#     text_detail = f"Compute for : {proj.detail()}  "

#     perf_data = proj.perf_table().round(1).to_dict(orient="records")

#     return fig, text_detail, perf_data


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)

# %%
