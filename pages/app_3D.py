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
    Project,
    plot_3d_cases,
    plot_3d_cases_risingangle,
    load_fish_kite,
)

import dash

dash.register_page(__name__)

from dash import dcc  # import dash_core_components as dcc   # from dash import dcc
from dash import html  # import dash_html_components as html  # from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import numpy as np
import copy
import pandas as pd
import os
import jsonpickle
from app_components_3d import *
from dash import ctx, dash_table, callback

__version__ = "1.0.3"

# %% Initial set up
data_folder = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data"
)  # we go up 2 folders

fk1_file = os.path.join(data_folder, "saved_fk1.json")
fk2_file = os.path.join(data_folder, "saved_fk2.json")

fk1 = FishKite.from_json(fk1_file)
fk2 = FishKite.from_json(fk2_file, classes=FishKite)
# fk1 = load_fish_kite(fk1_file)


proj = Project([fk1, fk2])
# %%"
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server
dfG = proj.create_df()

fig_rising_angle = plot_3d_cases_risingangle(dfG)
fig_all_pts = plot_3d_cases(dfG)
# df_table = proj.perf_table()
# CSS
tabs_styles = {
    "height": "30px",
    "padding-top": "2px",
}


# Layout

layout = dbc.Container(
    [
        dcc.Store(id="model_state", data={"need_update_sliders": True}),
        dcc.Store(id="graph_need_update", data={"need_update_sliders": True}),
        dcc.Store(id="isStartup", data=True),
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
                            "Internal model state",
                            color="info",
                            id="coordinates_button_3d",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Spinner(
                                            [html.Div(id="debug")],
                                            # html.P(id="my-output_3d"),
                                            color="primary",
                                        ),
                                        dbc.Spinner(
                                            [html.Div(id="debug2")],
                                            # html.P(id="my-output_3d"),
                                            color="primary",
                                        ),
                                    ],
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
                                        dbc.CardBody(
                                            create_polar_rising_sliders(),
                                            # dcc.Markdown("bidon")
                                        ),
                                        dcc.Graph(
                                            id="fig1_3d_rising_angle",
                                            figure=fig_rising_angle,
                                            # style={
                                            #     "position": "fixed",  # that imobilised the graph
                                            # },
                                        ),
                                    ],
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                ),
                                dcc.Tab(
                                    label="All Rising angle",
                                    children=[
                                        dbc.CardBody(
                                            create_polar_all_pts_sliders(),
                                            # dcc.Markdown("bidon")
                                        ),
                                        dcc.Graph(
                                            id="fig1_3d_all_pts",
                                            figure=fig_all_pts,
                                            # style={
                                            #     "position": "fixed",  # that imobilised the graph
                                            # },
                                        ),
                                    ],
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                ),
                            ],
                            parent_className="custom-tabs",
                            className="custom-tabs-container",
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


### Callback to update possible  rising angle according to truewind
@callback(
    Output("slider-rising_angle_polar", "marks"),
    [
        Input("3d_slider-wind_speed", "value"),
    ],
)
def update_possible_rising_angle(target_wind):
    possibility = list(
        dfG[dfG["true_wind_calculated_kt_rounded"] == target_wind][
            "rising_angle"
        ].unique()
    )

    return {int(i): str(i) for i in sorted(possibility)}


######################################################  GUI
### update slider

roots_update_slider = [
    "fk_name",
    "cable_strength",
    "3d_slider-kite_area",
    "3d_slider-kite_cl",
    "input_kite_flat_ratio",
    "input_kite_aspect_ratio",
    "input_kite_profildrag",
    "input_kite_parasitedrag",
    "input_kite_cable_length",
    "input_kite_cx_air",
    "3d_slider-fish_area",
    "3d_slider-fish_cl",
    "input_fish_flat_ratio",
    "input_fish_aspect_ratio",
    "input_fish_profildrag",
    "input_fish_parasitedrag",
    "input_fish_tip_depth",
    "input_fish_cable_length_unstreamline",
    "input_fish_cx_unstreamline",
    "input_fish_cable_length_streamline",
    "input_fish_cx_streamline",
]

list_outputs = [Output("model_state", "data")]
for id in [0, 1]:
    for field in roots_update_slider:
        list_outputs.append(Output(f"{field}_{id}", "value"))


@callback(
    list_outputs,
    Input("model_state", "data"),
)
def update_sliders(model_state):
    if model_state["need_update_sliders"]:
        output_to_send = []
        for id in [0, 1]:
            output_to_send.append(proj.lst_fishkite[id].name)
            output_to_send.append(proj.lst_fishkite[id].cable_strength)
            output_to_send.append(proj.lst_fishkite[id].kite._flat_area)
            output_to_send.append(list(proj.lst_fishkite[id].kite.cl_range.values()))
            output_to_send.append(proj.lst_fishkite[id].kite.flat_ratio)
            output_to_send.append(proj.lst_fishkite[id].kite.flat_aspect_ratio)
            output_to_send.append(proj.lst_fishkite[id].kite.profil_drag_coeff)
            output_to_send.append(proj.lst_fishkite[id].kite._parasite_drag_pct)
            output_to_send.append(proj.lst_fishkite[id].cable_length_kite)
            output_to_send.append(proj.lst_fishkite[id].cx_cable_air)
            output_to_send.append(proj.lst_fishkite[id].fish._flat_area)
            output_to_send.append(list(proj.lst_fishkite[id].fish.cl_range.values()))
            output_to_send.append(proj.lst_fishkite[id].fish.flat_ratio)
            output_to_send.append(proj.lst_fishkite[id].fish.flat_aspect_ratio)
            output_to_send.append(proj.lst_fishkite[id].fish.profil_drag_coeff)
            output_to_send.append(proj.lst_fishkite[id].fish._parasite_drag_pct)
            output_to_send.append(proj.lst_fishkite[id].tip_fish_depth)
            output_to_send.append(proj.lst_fishkite[id].cable_length_fish_unstreamline)
            output_to_send.append(proj.lst_fishkite[id].cx_cable_water_unstreamline)
            output_to_send.append(proj.lst_fishkite[id].cable_length_fish_streamline)
            output_to_send.append(proj.lst_fishkite[id].cx_cable_water_streamline)

        print(f" will update slider")

        return model_state, *output_to_send
    else:
        print(f" no update of the sliders")
        raise PreventUpdate


# startUP callback


@callback(
    [Output("isStartup", "data"), Output(f"3d_boolean_1", "on")],
    [
        Input("isStartup", "data"),
    ],
)
def Startup_call_back(data):
    if data:
        return False, False
    else:
        raise PreventUpdate


####################################################### PLOTS
### Callback to update polar selected  rising angle
@callback(
    Output("fig1_3d_rising_angle", "figure"),
    [
        Input("slider-rising_angle_polar", "value"),
        Input("bool_rising_angle_use_range", "on"),
        Input("3d_slider-wind_speed", "value"),
        Input("data_color_polar_rising", "value"),
        Input("data_symbol_polar_rising", "value"),
        Input("graph_need_update", "data"),
    ],
)
def update_polar_rising_angle(
    rising_angle, use_max_only, target_wind, color_data, symbol_data, _data
):
    if symbol_data == "None":
        symbol_data = None

    rising_low, rising_upper = rising_angle
    if use_max_only:
        rising_low = rising_upper

    return plot_3d_cases_risingangle(
        dfG,
        target_rising_angle_low=rising_low,
        target_rising_angle_upper=rising_upper,
        target_wind=target_wind,
        what=color_data,
        symbol=symbol_data,
    )


### Callback to update polar all  rising angle
@callback(
    Output("fig1_3d_all_pts", "figure"),
    [
        Input("3d_slider-wind_speed", "value"),
        Input("data_color_polar_all_pts", "value"),
        Input("graph_need_update", "data"),
    ],
)
def update_polar_all_pts(target_wind, color_data, jsonified_data):
    c = jsonified_data

    return plot_3d_cases(
        dfG,
        target_wind=target_wind,
        what=color_data,
    )


######################################################  DEBUG
### Callback to debug2
@callback(
    Output("debug2", "children"),
    [
        Input("model_state", "data"),
    ],
)
def update_polar_all_pts(model_state):
    return f"{model_state}"


######################################################  MODEL
# DF update callaback


roots_update_slider2 = [
    "fk_name",
    "cable_strength",
    "3d_slider-kite_area",
    "3d_slider-kite_cl",
    "input_kite_flat_ratio",
    "input_kite_aspect_ratio",
    "input_kite_profildrag",
    "input_kite_parasitedrag",
    "input_kite_cable_length",
    "input_kite_cx_air",
    "3d_slider-fish_area",
    "3d_slider-fish_cl",
    "input_fish_flat_ratio",
    "input_fish_aspect_ratio",
    "input_fish_profildrag",
    "input_fish_parasitedrag",
    "input_fish_tip_depth",
    "input_fish_cable_length_unstreamline",
    "input_fish_cx_unstreamline",
    "input_fish_cable_length_streamline",
    "input_fish_cx_streamline",
]

dict_input_update_model = {
    "all_inputs": {
        "general": {
            "wind_speed": Input("3d_slider-wind_speed", "value"),
        },
    }
}

for id in [0, 1]:
    d_i = {}
    d_i["3d_boolean"] = Input(f"3d_boolean_{id}", "on")
    for field in roots_update_slider2:
        d_i[field] = Input(f"{field}_{id}", "value")
    dict_input_update_model["all_inputs"][id] = d_i


@callback(
    [Output("graph_need_update", "data"), Output("debug", "children")],
    inputs=dict_input_update_model
    # {
    #     "all_inputs": {
    #         "general": {
    #             "wind_speed": Input("3d_slider-wind_speed", "value"),
    #         },
    #         0: {
    #             "3d_boolean": Input("3d_boolean_0", "on"),
    #             "3d_slider-kite_area": Input("3d_slider-kite_area_0", "value"),
    #             "kite_cl": Input("3d_slider-kite_cl_0", "value"),
    #             "input_kite_flat_ratio": Input("input_kite_flat_ratio_0", "value"),
    #             "3d_slider-fish_area": Input("3d_slider-fish_area_0", "value"),
    #             "fish_cl": Input("3d_slider-fish_cl_0", "value"),
    #         },
    #         1: {
    #             "3d_boolean": Input("3d_boolean_1", "on"),
    #             "3d_slider-kite_area": Input("3d_slider-kite_area_1", "value"),
    #             "kite_cl": Input("3d_slider-kite_cl_1", "value"),
    #             "input_kite_flat_ratio": Input("input_kite_flat_ratio_1", "value"),
    #             "3d_slider-fish_area": Input("3d_slider-fish_area_1", "value"),
    #             "fish_cl": Input("3d_slider-fish_cl_1", "value"),
    #         },
    #     }
    # },
)
def update(all_inputs):
    global dfG
    c = ctx.args_grouping.all_inputs
    case_list = []

    # fmt: off
    for idfk in [0, 1]:

        proj.lst_fishkite[idfk].wind_speed = c["general"]["wind_speed"]["value"]

        proj.lst_fishkite[idfk].name = c[idfk]["fk_name"]["value"]

        proj.lst_fishkite[idfk].kite._flat_area = c[idfk]["3d_slider-kite_area"]["value"]
        proj.lst_fishkite[idfk].kite.cl = c[idfk]["3d_slider-kite_cl"]["value"][1]
        proj.lst_fishkite[idfk].kite.cl_range["min"] = c[idfk]["3d_slider-kite_cl"]["value"][0]
        proj.lst_fishkite[idfk].kite.cl_range["max"] = c[idfk]["3d_slider-kite_cl"]["value"][1]
        proj.lst_fishkite[idfk].kite.flat_ratio = c[idfk]["input_kite_flat_ratio"]["value"]
        proj.lst_fishkite[idfk].kite.flat_aspect_ratio  = c[idfk]["input_kite_aspect_ratio"]["value"]
        proj.lst_fishkite[idfk].kite.profil_drag_coeff  = c[idfk]["input_kite_profildrag"]["value"]
        proj.lst_fishkite[idfk].kite._parasite_drag_pct = c[idfk]["input_kite_parasitedrag"]["value"]


        proj.lst_fishkite[idfk].fish._flat_area = c[idfk]["3d_slider-fish_area"]["value"]
        proj.lst_fishkite[idfk].fish.cl = c[idfk]["3d_slider-fish_cl"]["value"][1]
        proj.lst_fishkite[idfk].fish.cl_range["min"]    = c[idfk]["3d_slider-fish_cl"]["value"][0]
        proj.lst_fishkite[idfk].fish.cl_range["max"]    = c[idfk]["3d_slider-fish_cl"]["value"][1]
        proj.lst_fishkite[idfk].fish.flat_ratio = c[idfk]["input_fish_flat_ratio"]["value"]
        proj.lst_fishkite[idfk].fish.flat_aspect_ratio  = c[idfk]["input_fish_aspect_ratio"]["value"]
        proj.lst_fishkite[idfk].fish.profil_drag_coeff  = c[idfk]["input_fish_profildrag"]["value"]
        proj.lst_fishkite[idfk].fish._parasite_drag_pct = c[idfk]["input_fish_parasitedrag"]["value"]

        proj.lst_fishkite[idfk].fish._parasite_drag_pct = c[idfk]["input_fish_parasitedrag"]["value"]

        proj.lst_fishkite[idfk].cable_length_fish_unstreamline =c[idfk]["input_fish_cable_length_unstreamline"]["value"]
        proj.lst_fishkite[idfk].cable_length_fish_streamline =c[idfk]["input_fish_cable_length_streamline"]["value"]
        proj.lst_fishkite[idfk].cable_length_kite            =c[idfk]["input_kite_cable_length"]["value"]
        proj.lst_fishkite[idfk].cable_strength               =c[idfk]["cable_strength"]["value"]
        proj.lst_fishkite[idfk].cx_cable_water_unstreamline  =c[idfk]["input_fish_cx_unstreamline"]["value"]
        proj.lst_fishkite[idfk].cx_cable_water_streamline    =c[idfk]["input_fish_cx_streamline"]["value"]
        proj.lst_fishkite[idfk].cx_cable_air                 =c[idfk]["input_kite_cx_air"]["value"]
        proj.lst_fishkite[idfk].tip_fish_depth               =c[idfk]["input_fish_tip_depth"]["value"]

    # fmt: on
    if c[0]["3d_boolean"]["value"]:
        case_list.append(proj.lst_fishkite[0].name)
    if c[1]["3d_boolean"]["value"]:
        case_list.append(proj.lst_fishkite[1].name)

    # create data   # could be improve by not making the 2 fishkite if not needed

    dfall = proj.create_df()
    dfG = dfall[dfall["fk_name"].isin(case_list)]

    deb = f"updated df {dfG.shape} \n---\n={ c}"
    return True, deb


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)

# %%
