# plotly app

# %% import
#  http://127.0.0.1:8050/

from model import Deflector, FishKite, Project, plot_cases

import dash
from dash import dcc  # import dash_core_components as dcc   # from dash import dcc
from dash import html  # import dash_html_components as html  # from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import copy
import pandas as pd

from app_components import *
from dash import ctx, dash_table


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

fig = proj.plot()
df_table = proj.perf_table()


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                ## FishPy      V0.9.6
                """
                        )
                    ],
                    width=True,
                ),
            ],
            align="end",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Modify Operating Conditions", id="operating_button"
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    operating_slider_components,
                                )
                            ),
                            id="operating_collapse",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button("Modify FishKite_1 Parameters", id="shape1_button"),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    create_fk_sliders(0),
                                    # dcc.Markdown("bidon")
                                )
                            ),
                            id="shape1_collapse",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button("Modify FishKite_2 Parameters", id="shape2_button"),
                        html.Hr(),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    create_fk_sliders(1),
                                    # dcc.Markdown("bidon")
                                )
                            ),
                            id="shape2_collapse",
                            is_open=False,
                        ),
                        html.Hr(),
                        dbc.Button(
                            "Export data (txt format)",
                            id="coordinates_button",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    dbc.Spinner(
                                        html.P(id="my-output"),
                                        color="primary",
                                    ),
                                )
                            ),
                            id="coordinates_collapse",
                            is_open=False,
                        ),
                        html.Hr(),
                        # dcc.Markdown("##### Commands"),
                        # dbc.Button(
                        #     "Analyze",
                        #     id="analyze", color="primary", style={"margin": "5px"}),
                        html.Hr(),
                        dcc.Markdown("##### Aerodynamic Performance"),
                        dash_table.DataTable(
                            df_table.to_dict("records"),
                            [{"name": i, "id": i} for i in df_table.columns],
                            id="perf_table",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        # html.Div(id='display')
                        dcc.Graph(
                            id="fig1",
                            figure=fig,
                            style={
                                "position": "fixed",  # that imobilised the graph
                            },
                        )
                    ],
                    width=9,
                    align="start",
                ),
            ]
        ),
        dcc.Markdown(
            """
            
        To help the design
        """
        ),
    ],
    fluid=True,
)

## callback


### Callback to make shape parameters menu expand
@app.callback(
    Output("shape1_collapse", "is_open"),
    [Input("shape1_button", "n_clicks")],
    [State("shape1_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make shape parameters menu expand
@app.callback(
    Output("shape2_collapse", "is_open"),
    [Input("shape2_button", "n_clicks")],
    [State("shape2_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make operating parameters menu expand
@app.callback(
    Output("operating_collapse", "is_open"),
    [Input("operating_button", "n_clicks")],
    [State("operating_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to make coordinates menu expand
@app.callback(
    Output("coordinates_collapse", "is_open"),
    [Input("coordinates_button", "n_clicks")],
    [State("coordinates_collapse", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


### Callback to display L/D
@app.callback(
    Output("label-kite_eff_LD_0", "children"),
    Output("label-kite_eff_LD_1", "children"),
    Output("label-fish_eff_LD_0", "children"),
    Output("label-fish_eff_LD_1", "children"),
    [
        Input("slider-kite_efficiency_angle_0", "value"),
        Input("slider-kite_efficiency_angle_1", "value"),
        Input("slider-fish_efficiency_angle_0", "value"),
        Input("slider-fish_efficiency_angle_1", "value"),
    ],
)
def compute_LD_from_efficiency_angle(
    kite_ef_angle_0, kite_ef_angle_1, fish_ef_angle_0, fish_ef_angle_1
):
    LD_kite_0 = f"\t L/D= {1/np.arctan(np.radians(kite_ef_angle_0)):.2f}"
    LD_kite_1 = f"\t L/D= {1/np.arctan(np.radians(kite_ef_angle_1)):.2f}"
    LD_fish_0 = f"\t L/D= {1/np.arctan(np.radians(fish_ef_angle_0)):.2f}"
    LD_fish_1 = f"\t L/D= {1/np.arctan(np.radians(fish_ef_angle_1)):.2f}"
    return LD_kite_0, LD_kite_1, LD_fish_0, LD_fish_1


# call back sync FK
# TODO could be done using :Output dict grouping, for bidirectional
@app.callback(
    [
        Output("slider-rising_angle_1", "value"),
        Output("slider-kite_area_1", "value"),
        Output("slider-kite_cl_1", "value"),
        Output("slider-kite_efficiency_angle_1", "value"),
        Output("slider-fish_area_1", "value"),
        Output("slider-fish_cl_1", "value"),
        Output("slider-fish_efficiency_angle_1", "value"),
    ],
    inputs=[Input("copy_FK0toFK1", "n_clicks")],
)
def fk2_param_to_fk1(n_clicks):
    fk_from = 0

    if n_clicks:
        t = (
            proj.lst_fishkite[fk_from].rising_angle,
            proj.lst_fishkite[fk_from].kite.area,
            # cl kite list
            [
                proj.lst_fishkite[fk_from].kite.cl_range["min"],
                proj.lst_fishkite[fk_from].kite.cl,
                proj.lst_fishkite[fk_from].kite.cl_range["max"],
            ],
            proj.lst_fishkite[fk_from].kite.efficiency_angle,
            proj.lst_fishkite[fk_from].fish.area,
            # cl fish list
            [
                proj.lst_fishkite[fk_from].fish.cl_range["min"],
                proj.lst_fishkite[fk_from].fish.cl,
                proj.lst_fishkite[fk_from].fish.cl_range["max"],
            ],
            proj.lst_fishkite[fk_from].fish.efficiency_angle,
        )
    else:
        t = [dash.no_update] * 7

    return t


## main collaback
@app.callback(
    Output("fig1", "figure"),
    Output(component_id="my-output", component_property="children"),
    Output(component_id="perf_table", component_property="data"),
    inputs={
        "all_inputs": {
            "general": {
                "wind_speed": Input("slider-wind_speed", "value"),
                "bool_orthogrid": Input("bool_orthogrid", "on"),
                "bool_backgrdimg": Input("bool_backgrdimg", "on"),
                "bool_isospeed": Input("bool_isospeed", "on"),
                "bool_isoeft": Input("bool_isoeft", "on"),
                "graph_size": Input("slider-graph_size", "value"),
            },
            0: {
                "bool_fk": Input("boolean_0", "on"),
                "rising_angle": Input("slider-rising_angle_0", "value"),
                "kite_area": Input("slider-kite_area_0", "value"),
                "kite_cl": Input("slider-kite_cl_0", "value"),
                "kite_efficiency_angle": Input(
                    "slider-kite_efficiency_angle_0", "value"
                ),
                "fish_area": Input("slider-fish_area_0", "value"),
                "fish_cl": Input("slider-fish_cl_0", "value"),
                "fish_efficiency_angle": Input(
                    "slider-fish_efficiency_angle_0", "value"
                ),
            },
            1: {
                "bool_fk": Input("boolean_1", "on"),
                "rising_angle": Input("slider-rising_angle_1", "value"),
                "kite_area": Input("slider-kite_area_1", "value"),
                "kite_cl": Input("slider-kite_cl_1", "value"),
                "kite_efficiency_angle": Input(
                    "slider-kite_efficiency_angle_1", "value"
                ),
                "fish_area": Input("slider-fish_area_1", "value"),
                "fish_cl": Input("slider-fish_cl_1", "value"),
                "fish_efficiency_angle": Input(
                    "slider-fish_efficiency_angle_1", "value"
                ),
            },
        }
    },
)
def update(all_inputs):
    c = ctx.args_grouping.all_inputs

    bool_orthogrid = c["general"]["bool_orthogrid"]["value"]
    bool_backgrdimg = c["general"]["bool_backgrdimg"]["value"]
    bool_isospeed = c["general"]["bool_isospeed"]["value"]
    bool_isoeft = c["general"]["bool_isoeft"]["value"]

    proj.lst_fishkite[0].wind_speed = c["general"]["wind_speed"]["value"]
    proj.lst_fishkite[1].wind_speed = c["general"]["wind_speed"]["value"]

    proj.lst_fishkite[0].rising_angle = c[0]["rising_angle"]["value"]
    proj.lst_fishkite[0].kite.area = c[0]["kite_area"]["value"]
    proj.lst_fishkite[0].kite.cl = c[0]["kite_cl"]["value"][1]
    proj.lst_fishkite[0].kite.cl_range["min"] = c[0]["kite_cl"]["value"][0]
    proj.lst_fishkite[0].kite.cl_range["max"] = c[0]["kite_cl"]["value"][2]
    proj.lst_fishkite[0].kite.efficiency_angle = c[0]["kite_efficiency_angle"]["value"]
    proj.lst_fishkite[0].fish.area = c[0]["fish_area"]["value"]
    proj.lst_fishkite[0].fish.cl = c[0]["fish_cl"]["value"][1]
    proj.lst_fishkite[0].fish.cl_range["min"] = c[0]["fish_cl"]["value"][0]
    proj.lst_fishkite[0].fish.cl_range["max"] = c[0]["fish_cl"]["value"][2]
    proj.lst_fishkite[0].fish.efficiency_angle = c[0]["fish_efficiency_angle"]["value"]

    proj.lst_fishkite[1].rising_angle = c[1]["rising_angle"]["value"]
    proj.lst_fishkite[1].kite.area = c[1]["kite_area"]["value"]
    proj.lst_fishkite[1].kite.cl = c[1]["kite_cl"]["value"][1]
    proj.lst_fishkite[1].kite.cl_range["min"] = c[1]["kite_cl"]["value"][0]
    proj.lst_fishkite[1].kite.cl_range["max"] = c[1]["kite_cl"]["value"][2]
    proj.lst_fishkite[1].kite.efficiency_angle = c[1]["kite_efficiency_angle"]["value"]
    proj.lst_fishkite[1].fish.area = c[1]["fish_area"]["value"]
    proj.lst_fishkite[1].fish.cl = c[1]["fish_cl"]["value"][1]
    proj.lst_fishkite[1].fish.cl_range["min"] = c[1]["fish_cl"]["value"][0]
    proj.lst_fishkite[1].fish.cl_range["max"] = c[1]["fish_cl"]["value"][2]
    proj.lst_fishkite[1].fish.efficiency_angle = c[1]["fish_efficiency_angle"]["value"]

    case_to_plot = []

    if c[0]["bool_fk"]["value"]:
        case_to_plot.append(proj.lst_fishkite[0])
    if c[1]["bool_fk"]["value"]:
        case_to_plot.append(proj.lst_fishkite[1])

    fig = plot_cases(
        list_of_cases=case_to_plot,
        draw_ortho_grid=bool_orthogrid,
        draw_iso_speed=bool_isospeed,
        draw_iso_eft=bool_isoeft,
        add_background_image=bool_backgrdimg,
        height_size=c["general"]["graph_size"]["value"],
    )

    text_detail = f"Compute for : {proj.detail()}  "

    perf_data = proj.perf_table().round(2).to_dict(orient="records")

    return fig, text_detail, perf_data


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)

# %%
