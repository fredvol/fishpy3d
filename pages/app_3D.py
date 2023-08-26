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

dcc.Store(id="model_state", data={"need_update": True}),

# Layout

layout = dbc.Container(
    [
        dcc.Store(id="model_state", data={"need_update": True}),
        dbc.Row(
            [
                dcc.Store(id="dataStore", storage_type="memory"),
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


### update slider


@callback(
    [Output("model_state", "data"), Output("input_kite_flat_ratio_0", "value")],
    Input("model_state", "data"),
)
def filter_countries(model_state):
    if model_state["need_update"]:
        flat_ratio_0 = proj.lst_fishkite[0].kite.flat_ratio
        print(f" will update flat ratio : {flat_ratio_0}")
        model_state["need_update"] = False
        return model_state, flat_ratio_0
    else:
        print(f" no update : {flat_ratio_0}")
        raise PreventUpdate


### Callback to update polar selected  rising angle
@callback(
    Output("fig1_3d_rising_angle", "figure"),
    [
        Input("slider-rising_angle_polar", "value"),
        Input("bool_rising_angle_use_range", "on"),
        Input("3d_slider-wind_speed", "value"),
        Input("data_color_polar_rising", "value"),
        Input("data_symbol_polar_rising", "value"),
        Input("dataStore", "data"),
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
        Input("dataStore", "data"),
    ],
)
def update_polar_all_pts(target_wind, color_data, jsonified_data):
    c = jsonified_data

    return plot_3d_cases(
        dfG,
        target_wind=target_wind,
        what=color_data,
    )


### Callback to debug2
@callback(
    Output("debug2", "children"),
    [
        Input("dataStore", "data"),
    ],
)
def update_polar_all_pts(_data):
    global dfG
    df_max = dfG.groupby("fk_name")["vmg_x"].max()
    result = ""
    for name in dfG["fk_name"].unique():
        result += f"Max_{name}: {df_max[name]} "
    return result


# DF update callaback
@callback(
    [Output("dataStore", "data"), Output("debug", "children")],
    inputs={
        "all_inputs": {
            "general": {
                "wind_speed": Input("3d_slider-wind_speed", "value"),
            },
            0: {
                "bool_fk": Input("3d_boolean_0", "on"),
                "kite_area": Input("3d_slider-kite_area_0", "value"),
                "kite_cl": Input("3d_slider-kite_cl_0", "value"),
                "fish_area": Input("3d_slider-fish_area_0", "value"),
                "fish_cl": Input("3d_slider-fish_cl_0", "value"),
            },
            1: {
                "bool_fk": Input("3d_boolean_1", "on"),
                "kite_area": Input("3d_slider-kite_area_1", "value"),
                "kite_cl": Input("3d_slider-kite_cl_1", "value"),
                "fish_area": Input("3d_slider-fish_area_1", "value"),
                "fish_cl": Input("3d_slider-fish_cl_1", "value"),
            },
        }
    },
    # State=State("dataStore", "data"),
)
def update(all_inputs):
    global dfG
    c = ctx.args_grouping.all_inputs
    case_list = []

    if c[0]["bool_fk"]["value"]:
        case_list.append(proj.lst_fishkite[0].name)
    if c[1]["bool_fk"]["value"]:
        case_list.append(proj.lst_fishkite[1].name)

    proj.lst_fishkite[0].wind_speed = c["general"]["wind_speed"]["value"]
    proj.lst_fishkite[1].wind_speed = c["general"]["wind_speed"]["value"]

    proj.lst_fishkite[0].kite._flat_area = c[0]["kite_area"]["value"]
    proj.lst_fishkite[0].kite.cl = c[0]["kite_cl"]["value"][1]
    proj.lst_fishkite[0].kite.cl_range["min"] = c[0]["kite_cl"]["value"][0]
    proj.lst_fishkite[0].kite.cl_range["max"] = c[0]["kite_cl"]["value"][1]

    proj.lst_fishkite[1].kite._flat_area = c[1]["kite_area"]["value"]
    proj.lst_fishkite[1].kite.cl = c[1]["kite_cl"]["value"][1]
    proj.lst_fishkite[1].kite.cl_range["min"] = c[1]["kite_cl"]["value"][0]
    proj.lst_fishkite[1].kite.cl_range["max"] = c[1]["kite_cl"]["value"][1]

    dfall = proj.create_df()
    dfG = dfall[dfall["fk_name"].isin(case_list)]

    deb = f"updated df {dfG.shape} \n---\n={ c}"
    return c, deb


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)

# %%
