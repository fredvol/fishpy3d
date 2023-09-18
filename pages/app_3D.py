# plotly app

# %% import
#  http://127.0.0.1:8050/
# git push heroku main
# https://fishpy2d-9143325b137e.herokuapp.com/

# imports
import base64
from copy import deepcopy
import io
from model_3d import (
    Deflector,
    FishKite,
    Pilot,
    Project,
    plot_3d_cases,
    plot_3d_cases_risingangle,
    perf_table,
    perf_table_general,
)
from fish_plot_3d import plot_side_view
import dash

dash.register_page(__name__)
import json
from dash import dcc  # import dash_core_components as dcc   # from dash import dcc
from dash import html  # import dash_html_components as html  # from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


import os
from app_components_3d import *
from dash import ctx, dash_table, callback

__version__ = "2.1.5"
print("Version: ", __version__)
print("The browser will try to start automatically.")
print("(few seconds for the initialisation of the browser can be needed)")

print("--")

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

fig_rising_angle = plot_3d_cases_risingangle(dfG[dfG["fk_name"] == fk1.name])
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Modify Operating Conditions",
                                        color="danger",
                                        id="operating_button_3d",
                                    )
                                ),
                                dbc.Col(
                                    dcc.Loading(
                                        id="loading-1",
                                        type="default",
                                        children=[
                                            html.Br(),
                                            html.Div(
                                                "State: ready ",
                                                id="load",
                                                className="loading-labels",
                                            ),
                                            html.Div(" ", id="load_model"),
                                            html.Div(" ", id="load_graph"),
                                        ],
                                    ),
                                    width={"size": 3, "order": 5},
                                ),
                            ]
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
                                            [html.Div(id="df_info")],
                                            color="primary",
                                        ),
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Download Excel (very long!)",
                                                    id="btn_xlsx",
                                                ),
                                                dcc.Download(
                                                    id="download-dataframe-xlsx"
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        html.Div("Model state:"),
                                        dbc.Spinner(
                                            [html.Div(id="debug")],
                                            # html.P(id="my-output_3d"),
                                            color="primary",
                                        ),
                                        html.Hr(),
                                        html.Div("Debug:"),
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
                        dcc.Markdown(
                            "##### Numerical Results: (*'no failure' OP only*)"
                        ),
                        dcc.Markdown(" * At selected wind:"),
                        dash_table.DataTable(
                            id="perf_table_3d_selectedW",
                        ),
                        dcc.Markdown(" * All True winds:"),
                        dash_table.DataTable(
                            id="perf_table_3d_allW",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="selected Rising angle(s)",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="fig1_3d_rising_angle",
                                                            # figure=fig_rising_angle,
                                                            # style={
                                                            #     "position": "fixed",  # that imobilised the graph
                                                            # },
                                                        ),
                                                        dcc.Graph(
                                                            id="fig1_3d_side_view",
                                                            # style={
                                                            #     "position": "fixed",  # that imobilised the graph
                                                            # },
                                                        ),
                                                    ],
                                                    width=9,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.H5(
                                                                    "Graph control:"
                                                                ),
                                                                dbc.CardBody(
                                                                    create_polar_rising_sliders(),
                                                                    # dcc.Markdown("bidon")
                                                                ),
                                                            ]
                                                        ),
                                                        #
                                                        html.Br(),
                                                        dbc.Stack(
                                                            [
                                                                html.H5(
                                                                    "OP specific data:"
                                                                ),
                                                                dash_table.DataTable(
                                                                    id="click_data_table",
                                                                    export_format="csv",
                                                                    editable=True,
                                                                    style_cell_conditional=[
                                                                        {
                                                                            "if": {
                                                                                "column_id": "Unit"
                                                                            },
                                                                            "textAlign": "left",
                                                                        }
                                                                    ],
                                                                    style_data={
                                                                        "color": "black",
                                                                        "backgroundColor": "white",
                                                                        "font-size": "0.7em",
                                                                    },
                                                                    style_data_conditional=[
                                                                        {
                                                                            "if": {
                                                                                "row_index": "odd"
                                                                            },
                                                                            "backgroundColor": "rgb(220, 220, 220)",
                                                                        }
                                                                    ],
                                                                    style_header={
                                                                        "backgroundColor": "rgb(210, 210, 210)",
                                                                        "color": "black",
                                                                        "fontWeight": "bold",
                                                                    },
                                                                    # style_table={
                                                                    #     "height": "800px",
                                                                    #     "overflowY": "auto",
                                                                    # },
                                                                    css=[
                                                                        {
                                                                            "selector": ".dash-spreadsheet tr",
                                                                            "rule": "height: 10px;",
                                                                        },
                                                                        {
                                                                            "selector": "tr:first-child",
                                                                            "rule": "display: none",
                                                                        },
                                                                        {
                                                                            "selector": ".export",
                                                                            "rule": "position:absolute;right:-15px;bottom:-30px",
                                                                        },
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                            ]
                                        )
                                    ],
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                ),
                                # dcc.Tab(   # extra tabs for all rising angles
                                #     label="All Rising angle",
                                #     children=[
                                #         dbc.Row(
                                #             [
                                #                 dbc.Col(
                                #                     [
                                #                         dbc.CardBody(
                                #                             create_polar_all_pts_sliders(),
                                #                             # dcc.Markdown("bidon")
                                #                         ),
                                #                         dcc.Graph(
                                #                             id="fig1_3d_all_pts",
                                #                             figure=fig_all_pts,
                                #                             # style={
                                #                             #     "position": "fixed",  # that imobilised the graph
                                #                             # },
                                #                         ),
                                #                     ]
                                #                 ),
                                #                 dbc.Col(
                                #                     [
                                #                         html.Div(
                                #                             "One of three columns"
                                #                         ),
                                #                     ]
                                #                 ),
                                #             ]
                                #         )
                                #     ],
                                #     className="custom-tab",
                                #     selected_className="custom-tab--selected",
                                # ),
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
        html.Div(
            [
                dbc.Button(
                    "Hypothesis and legend",
                    id="open-offcanvas",
                    n_clicks=0,
                    color="info",
                ),
                dbc.Offcanvas(
                    dcc.Markdown(
                        """
        **Hypothesis:**

            * Extra angle > 1 deg
            * Wind Speed input, limited to even integers. Polar graph winds = Wind speed input +/-1kt.
            * Cable drag coeficient (air or water) independant from speed. 
            * Induced drag calculation with Oswald efficency = 1
            * fish cavitated if water speed > 40kt or Water pressure > 5000daN/m²
            * cable break if fish_total_force > cable_strength
            * Aspect Ratio input: Wing Aspect Ratio if it was unroll on a flat surface. Span*Span/Aera. 
            * Pilot Mass: mass of everything standing in the air (pilot, harness, lines, paraglider)

        **Legend:**

            * OP = Operation point
            * fk = Fish-Kite.  Fk name: do not put same name for fk1 and fk2. 
            * Fluid ratio = Apparent Water Speed / Apparent wind Speed
            * efficiency in angle = ATAN (Drag/Lift)
            * Total efficiency (degrees) = fish efficiency + kite effiency
            * positive VMG_y = VMG upwind. Negative VMG_y = VMG downwind. 
            * Flat area input = aera if the wing is unrolled on a flat surface. 
            * parasite drag (fish or kite) input: in m² as ratio of flat area. This drag include anything that is not profile drag or induced drag, for exemple all fish or kite extra structures, like load bearing distribution lines or struts. 
            * cables: 
                - streamline and unstreamline = cable between fish and pilot
                - length kite = cable between pilot and kite. 
            * Lift Ceoficient range input = maximum and minimum reachable Cl for a given fish or kite. Should be set to identical values if not steerable. 
            * Valid Points = point of sail with no cavitation and no cable break. 
            * Data scroll menu = data type specifically displayed on polar graph with color code and hover legend.
            * Symbol scroll menu = differentiation for OP with câble break, cavitation, etc. 
            * Simplify OP: OP along a linear progressive change of Cl for both kite and fish. """
                    ),
                    id="offcanvas",
                    title="Hypothesis and legend",
                    placement="bottom",
                    is_open=False,
                    style={"height": 450},
                ),
            ]
        ),
        dcc.Markdown(
            f"""
        ......................................  
        **Version:** {__version__}
        """
        ),
    ],
    fluid=True,
)

## callback


@callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


### Callback to parameters menus expand
@callback(
    Output("shape1_collapse_3d", "is_open"),
    [Input("shape1_button_3d", "n_clicks")],
    [State("shape1_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("shape2_collapse_3d", "is_open"),
    [Input("shape2_button_3d", "n_clicks")],
    [State("shape2_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("operating_collapse_3d", "is_open"),
    [Input("operating_button_3d", "n_clicks")],
    [State("operating_collapse_3d", "is_open")],
)
def toggle_shape_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


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


###################################################### Click reaction


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


summary_table_fields = {
    "kite_cl": " ",
    "fish_cl": " ",
    "rising_angle": "°",
    "extra_angle": "°",
    "simplify": " ",
    "fish_center_depth": "m",
    "cable_length_in_water": "m",
    "cable_length_in_water_streamline": "m",
    "cable_length_in_water_unstreamline": "m",
    "cable_water_drag": "m²",
    "cable_length_in_air": "m",
    "cable_air_drag": "m²",
    "fish_lift": "N",
    "kite_lift": "N",
    "fish_induced_drag": "m²",
    "kite_induced_drag": "m²",
    "total_water_drag": "m²",
    "total_air_drag": "m²",
    "fish_total_force": "N",
    "kite_total_force": "N",
    "proj_efficiency_water_LD": " ",
    "proj_efficiency_air_LD": " ",
    "y_pilot": "m",
    "z_pilot": "m",
    "y_kite": "m",
    "z_kite": "m",
    "apparent_water_kt": "kt",
    "apparent_wind_kt": "kt",
    "true_wind_calculated_kt": "kt",
    "vmg_x_kt": "kt",
    "vmg_y_kt": "kt",
    "cable_strength_margin": " ",
    "cavitation": " ",
    "fk_name": " ",
    "failure": " ",
    "indexG": " ",
}

shorter_name = {
    "cable_length_in_water_streamline": "cable_lngth_water_stream",
    "cable_length_in_water_unstreamline": "cable_lngth_water_unstream",
}


@callback(
    [
        Output(component_id="click_data_table", component_property="data"),
        Output(component_id="click_data_table", component_property="columns"),
        Output("fig1_3d_side_view", "figure"),
    ],
    Input("fig1_3d_rising_angle", "clickData"),
    prevent_initial_call=True,
)
def display_click_data(clickData):
    df_index = clickData["points"][0]["customdata"][-1]
    df_OP = dfG[(dfG["indexG"] == df_index)]
    df_click_select = df_OP[summary_table_fields.keys()]
    # Identify numeric columns
    numeric_cols = df_click_select.select_dtypes(include=[float, int]).columns

    # Format numeric columns with n digits
    df_click_select[numeric_cols] = df_click_select[numeric_cols].round(4)

    df_click = df_click_select.reset_index().T.reset_index()

    # rename
    df_click["index"] = df_click["index"].replace(
        shorter_name
    )  # map wouldbe much faster,less lisible: df_click["index"] = df_click["index"].map(shorter_name).fillna( df_click["index"])

    # add units:
    df_click["Unit"] = df_click["index"].map(summary_table_fields)

    # conver to dash format
    columns = [{"name": str(col), "id": str(col)} for col in df_click.columns]
    data = df_click.to_dict(orient="records")

    # side view
    fig_side = plot_side_view(df_OP.to_dict("records")[0], proj.lst_fishkite[0])

    return data, columns, fig_side


###################################################### DOWNLOAD


@callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn_xlsx", "n_clicks"),
    prevent_initial_call=True,
)
def download_bigDF(n_clicks):
    return dcc.send_data_frame(
        dfG.to_excel, "df_general.xlsx", sheet_name="Sheet_name_1"
    )


@callback(
    Output("download-fk1", "data"),
    Input("export_fk0", "n_clicks"),
    Input("export_fk1", "n_clicks"),
    prevent_initial_call=True,
)
def download_fk(btn_fk0, btn_fk1):
    button_clicked = ctx.triggered_id
    if button_clicked == "export_fk0":
        return dict(content=fk1.to_json_str(), filename=f"exported_{fk1.name}.txt")
    elif button_clicked == "export_fk1":
        return dict(content=fk2.to_json_str(), filename=f"exported_{fk2.name}.txt")
    else:
        print("Impossible to export : button_clicked", button_clicked)
        raise PreventUpdate


####### copy other
# @callback(
#     Output("model_state", "data"),
#     Input("copy_fk0", "n_clicks"),
#     Input("copy_fk1", "n_clicks"),
#     prevent_initial_call=True,
# )
# def download_fk(btn_fk0, btn_fk1):
#     button_clicked = ctx.triggered_id
#     if button_clicked == "copy_fk0":
#         old_name = proj.lst_fishkite[0].name

#         proj.lst_fishkite[1] = deepcopy(proj.lst_fishkite[0])
#         proj.lst_fishkite[1].name = old_name
#         return {"need_update_sliders": True}
#     elif button_clicked == "copy_fk1":
#         old_name = proj.lst_fishkite[1].name

#         proj.lst_fishkite[0] = deepcopy(proj.lst_fishkite[1])
#         proj.lst_fishkite[0].name = old_name
#     else:
#         print("Impossible to copy : button_clicked", button_clicked)
#         raise PreventUpdate


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
    "cable_strength_mm2",
    "input_pilot_mass",
    "input_pilot_drag",
]


list_outputs = [Output("model_state", "data")]
for id in [0, 1]:
    for field in roots_update_slider:
        list_outputs.append(Output(f"{field}_{id}", "value"))


@callback(
    list_outputs,
    Input("model_state", "data"),
    Input("inport_fk0", "contents"),
    Input("inport_fk1", "contents"),
)
def update_sliders(model_state, data_import0, data_import1):
    button_clicked = ctx.triggered_id
    # print(f" triger by {button_clicked}")

    if button_clicked == "inport_fk0":
        content_type, content_string = data_import0.split(",")
        decoded = base64.b64decode(content_string)
        proj.lst_fishkite[0] = FishKite.from_json_str(decoded)

    if button_clicked == "inport_fk1":
        content_type, content_string = data_import1.split(",")
        decoded = base64.b64decode(content_string)
        proj.lst_fishkite[1] = FishKite.from_json_str(decoded)

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
            output_to_send.append(proj.lst_fishkite[id].cable_strength_mm2)
            output_to_send.append(proj.lst_fishkite[id].pilot.mass)
            output_to_send.append(proj.lst_fishkite[id].pilot.pilot_drag)

        # print(f" will update slider")

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
    Output("load_graph", "children"),
    [
        Input("slider-rising_angle_polar", "value"),
        Input("bool_rising_angle_use_range", "value"),
        Input("3d_slider-wind_speed", "value"),
        Input("data_color_polar_rising", "value"),
        Input("data_symbol_polar_rising", "value"),
        Input("graph_need_update", "data"),
        Input("3d_bool_orthogrid", "on"),
        Input("3d_bool_isospeed", "on"),
        Input("3d_bool_isoeft", "on"),
        Input("3d_bool_isofluid", "on"),
        Input("3d_slider-graph_size", "value"),
        Input("bool_validOP_only", "value"),
        Input("bool_draw_simplify_OP", "on"),
    ],
)
def update_polar_rising_angle(
    rising_angle,
    use_range,
    target_wind,
    color_data,
    symbol_data,
    _data,
    bool_orthogrid,
    bool_isospeed,
    bool_isoeft,
    bool_isofluid,
    graph_size,
    bool_ValidOP_only,
    bool_SimplifiedOP,
):
    if symbol_data == "None":
        symbol_data = None

    rising_low, rising_upper = rising_angle
    if not use_range:
        rising_low = rising_upper

    if bool_ValidOP_only:
        df_rising_angle = dfG[dfG["isValid"]]
    else:
        df_rising_angle = dfG

    empty_for_load = " "

    return (
        plot_3d_cases_risingangle(
            df_rising_angle,
            target_rising_angle_low=rising_low,
            target_rising_angle_upper=rising_upper,
            target_wind=target_wind,
            what=color_data,
            symbol=symbol_data,
            draw_ortho_grid=bool_orthogrid,
            draw_iso_speed=bool_isospeed,
            draw_iso_eft=bool_isoeft,
            draw_iso_fluid=bool_isofluid,
            height_size=graph_size,
            draw_simplify_OP=bool_SimplifiedOP,
        ),
        empty_for_load,
    )


### Callback to update polar all  rising angle
# @callback(
#     Output("fig1_3d_all_pts", "figure"),
#     [
#         Input("3d_slider-wind_speed", "value"),
#         Input("data_color_polar_all_pts", "value"),
#         Input("graph_need_update", "data"),
#     ],
# )
# def update_polar_all_pts(target_wind, color_data, jsonified_data):
#     c = jsonified_data

#     return plot_3d_cases(
#         dfG,
#         target_wind=target_wind,
#         what=color_data,
#     )


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


# roots_update_slider2 = [
#     "fk_name",
#     "cable_strength",
#     "3d_slider-kite_area",
#     "3d_slider-kite_cl",
#     "input_kite_flat_ratio",
#     "input_kite_aspect_ratio",
#     "input_kite_profildrag",
#     "input_kite_parasitedrag",
#     "input_kite_cable_length",
#     "input_kite_cx_air",
#     "3d_slider-fish_area",
#     "3d_slider-fish_cl",
#     "input_fish_flat_ratio",
#     "input_fish_aspect_ratio",
#     "input_fish_profildrag",
#     "input_fish_parasitedrag",
#     "input_fish_tip_depth",
#     "input_fish_cable_length_unstreamline",
#     "input_fish_cx_unstreamline",
#     "input_fish_cable_length_streamline",
#     "input_fish_cx_streamline",
# ]

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
    for field in roots_update_slider:
        d_i[field] = Input(f"{field}_{id}", "value")
    dict_input_update_model["all_inputs"][id] = d_i


@callback(
    [
        Output("graph_need_update", "data"),
        Output("df_info", "children"),
        Output("debug", "children"),
        Output("perf_table_3d_selectedW", "columns"),
        Output("perf_table_3d_selectedW", "data"),
        Output("perf_table_3d_allW", "columns"),
        Output("perf_table_3d_allW", "data"),
        Output("load_model", "children"),
    ],
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
        proj.lst_fishkite[idfk].cable_strength_mm2           =c[idfk]["cable_strength_mm2"]["value"]
        proj.lst_fishkite[idfk].pilot.mass                   =c[idfk]["input_pilot_mass"]["value"]
        proj.lst_fishkite[idfk].pilot.pilot_drag             =c[idfk]["input_pilot_drag"]["value"]

    # fmt: on
    if c[0]["3d_boolean"]["value"]:
        case_list.append(proj.lst_fishkite[0].name)
    if c[1]["3d_boolean"]["value"]:
        case_list.append(proj.lst_fishkite[1].name)

    # create data   # could be improve by not making the 2 fishkite if not needed

    dfall = proj.create_df()
    dfG = dfall[dfall["fk_name"].isin(case_list)]

    info_df = f"Data Table: rows: {dfG.shape[0]} :cols: {dfG.shape[1]} , {sum(dfG.memory_usage())/1e6} mb"

    model_state = proj.to_json_str()
    # perf data _selectedWind
    target_wind = c["general"]["wind_speed"]["value"]
    df_perf = perf_table(dfG, target_wind=target_wind).round(2).reset_index()

    perf_columns_selectedWind = [
        {"name": str(i), "id": str(i)} for i in df_perf.columns
    ]
    perf_data_selectedWind = df_perf.to_dict("records")

    # perf general
    df_perf_general = perf_table_general(dfG, proj).T.reset_index()
    perf_columns_general = [
        {"name": str(i), "id": str(i)} for i in df_perf_general.columns
    ]
    perf_data_general = df_perf_general.to_dict("records")
    empty_for_loading = " "
    return (
        True,
        info_df,
        model_state,
        perf_columns_selectedWind,
        perf_data_selectedWind,
        perf_columns_general,
        perf_data_general,
        empty_for_loading,
    )  # perf_columns, perf_data


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)

# %%
