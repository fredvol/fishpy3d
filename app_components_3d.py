from dash import dcc
from dash import html
import dash_daq as daq
import numpy as np
import dash_bootstrap_components as dbc

### Operating things
operating_slider_components = [
    html.Div(
        [
            html.Label("Graph size:"),
            dcc.Slider(
                id=f"3d_slider-graph_size",
                min=500,
                max=1000,
                step=50,
                value=800,
                marks={i: f"{i} px" for i in range(500, 1100, 100)},
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
            ),
        ],
    ),
    html.Label("Wind speed [knots]"),
    dcc.Slider(
        id=f"3d_slider-wind_speed",
        min=1,
        max=40,
        step=1,
        value=20,
        marks={i: str(i) for i in range(0, 40, 5)},
        tooltip={
            "placement": "bottom",
            "always_visible": True,
        },
    ),
    daq.BooleanSwitch(
        id="3d_bool_orthogrid", on=True, label="Ortho grid", labelPosition="right"
    ),
    # daq.BooleanSwitch(
    #     id="bool_backgrdimg", on=False, label="Background image", labelPosition="right"
    # ),
    daq.BooleanSwitch(
        id="3d_bool_isospeed", on=True, label="Iso speed", labelPosition="right"
    ),
    daq.BooleanSwitch(
        id="3d_bool_isoeft",
        on=True,
        label="Iso efficiency total",
        labelPosition="right",
    ),
    daq.BooleanSwitch(
        id="3d_bool_isofluid", on=True, label="Iso fluid ratio", labelPosition="right"
    ),
    # dbc.Button(
    #     "copy parameters : Fk1 -> Fk2",
    #     color="secondary",
    #     size="sm",
    #     id="copy_FK0toFK1",
    # ),
]

### FishKite


def create_polar_rising_sliders():
    s = html.Div(
        dbc.Col(
            [
                html.Label("rising Angle [deg]"),
                dcc.RangeSlider(
                    id=f"slider-rising_angle_polar",
                    min=1,
                    max=90,
                    step=90,
                    value=[20, 30],
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 90, 5)},
                    pushable=1,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                daq.BooleanSwitch(
                    id="bool_rising_angle_use_range",
                    on=False,
                    label="Use max range only:",
                    labelPosition="left",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Data:"),
                                dcc.Dropdown(
                                    [
                                        "kite_cl",
                                        "fish_cl",
                                        "rising_angle",
                                        "extra_angle",
                                        "fish_center_depth",
                                        "cable_length_in_water",
                                        "cable_water_drag",
                                        "cable_air_drag",
                                        "fish_lift",
                                        "kite_lift",
                                        "total_water_drag",
                                        "total_air_drag",
                                        "fish_total_force",
                                        "kite_total_force",
                                        "z_pilot",
                                        "apparent_watter_kt",
                                        "apparent_wind_kt",
                                        "true_wind_calculated_kt",
                                        "vmg_x_kt",
                                        "vmg_y_kt",
                                        "cable_strength_margin",
                                        "fk_name",
                                    ],
                                    "extra_angle",
                                    id="data_color_polar_rising",
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.Label("Symbol:"),
                                dcc.Dropdown(
                                    ["None", "cable_break", "cavitation", "fk_name"],
                                    None,
                                    id="data_symbol_polar_rising",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    )
    return s


def create_polar_all_pts_sliders():
    s = html.Div(
        dbc.Col(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Data:"),
                                dcc.Dropdown(
                                    [
                                        "extra_angle",
                                        "fish_total_force",
                                        "apparent_watter_ms",
                                        "fk_name",
                                    ],
                                    "extra_angle",
                                    id="data_color_polar_all_pts",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    )
    return s


def create_fk_sliders(id):
    s = (
        html.Div(
            [
                html.Tr(
                    [
                        dbc.Stack(
                            [
                                html.H5(f"FishKite {id +1}"),
                                daq.BooleanSwitch(id=f"3d_boolean_{id}", on=True),
                                dcc.Upload(
                                    dbc.Button(
                                        "Inport",
                                        size="sm",
                                        color="secondary",
                                    ),
                                    id=f"inport_fk{id}",
                                ),
                                dbc.Button(
                                    "Export",
                                    id=f"export_fk{id}",
                                    size="sm",
                                    color="secondary",
                                ),
                                dcc.Download(id="download-fk1"),
                            ],
                            direction="horizontal",
                            gap=1,
                        ),
                        dbc.Stack(
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Name:",
                                        ),
                                        dbc.Input(type="text", id=f"fk_name_{id}"),
                                        dbc.Label(
                                            "cable strength:",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"cable_strength_{id}",
                                            min=100,
                                            max=2000,
                                            step=1,
                                        ),
                                        dbc.Tooltip(
                                            "unit: DaN ",
                                            target=f"cable_strength_{id}",
                                        ),
                                    ],
                                    direction="horizontal",
                                    gap=1,
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Stack(  # start Kite stack
                    [
                        html.H6(html.B(f"Kite:")),
                        html.Label("Kite Area [m2]"),
                        dcc.Slider(
                            id=f"3d_slider-kite_area_{id}",
                            min=1,
                            max=100,
                            step=1,
                            value=22,
                            updatemode="drag",
                            marks={i: str(i) for i in range(0, 105, 5)},
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        html.Label("Kite Force Coeficient range and OP"),
                        dcc.RangeSlider(
                            id=f"3d_slider-kite_cl_{id}",
                            min=0,
                            max=1.5,
                            step=0.05,
                            value=[0.5, 1],
                            updatemode="drag",
                            marks={i: str(i) for i in [0, 0.5, 1, 1.5]},
                            pushable=0,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        dbc.Stack(  # Kite flat ratio , aspect ratio
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "flat ratio:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_flat_ratio_{id}",
                                            min=0.5,
                                            max=1,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Aspect ratio:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_aspect_ratio_{id}",
                                            min=1,
                                            max=20,
                                            step=0.1,
                                            className="custom-inputs",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                        dbc.Stack(  # kite drag
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "profile drag coeff:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_profildrag_{id}",
                                            min=0.01,
                                            max=0.1,
                                            step=0.001,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m² ",
                                            target=f"input_kite_profildrag_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Parasite drag pct:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_parasitedrag_{id}",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: percentage of flat area ",
                                            target=f"input_kite_parasitedrag_{id}",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                        dbc.Label("Cable:"),
                        dbc.Stack(  # Kite Cable
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "length:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_cable_length_{id}",
                                            min=0,
                                            max=100,
                                            step=0.5,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m ",
                                            target=f"input_kite_cable_length_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "CX air:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_kite_cx_air_{id}",
                                            min=0.5,
                                            max=2,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "no unit ",
                                            target=f"input_kite_cx_air_{id}",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                    ],
                    gap=1,
                ),  # end KITE stack
                html.Br(),
                dbc.Stack(  # start Fish stack
                    [
                        html.H6(html.B(f"Fish:")),
                        html.Label("Fish Area [m2]"),
                        dcc.Slider(
                            id=f"3d_slider-fish_area_{id}",
                            min=0.01,
                            max=1,
                            step=0.01,
                            value=0.1,
                            updatemode="drag",
                            marks={i: f"{i:.2f}" for i in np.arange(0, 1.1, 0.1)},
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        html.Label("Fish Force Coeficient range and OP"),
                        dcc.RangeSlider(
                            id=f"3d_slider-fish_cl_{id}",
                            min=0.1,
                            max=1,
                            step=0.05,
                            value=[0.25, 0.5],
                            pushable=0,
                            updatemode="drag",
                            marks={i: str(i) for i in [0, 0.5, 1]},
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        dbc.Stack(  # Fish flat ratio , aspect ratio
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "flat ratio:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_flat_ratio_{id}",
                                            min=0.1,
                                            max=1,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Aspect ratio:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_aspect_ratio_{id}",
                                            min=0,
                                            max=50,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                        dbc.Stack(  # Fish drag and flat ratio
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "profile drag coeff:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_profildrag_{id}",
                                            min=0.005,
                                            max=0.05,
                                            step=0.005,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m² ",
                                            target=f"input_fish_profildrag_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Parasite drag pct:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_parasitedrag_{id}",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: percentage of flat area ",
                                            target=f"input_fish_parasitedrag_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "Tip depth:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_tip_depth_{id}",
                                            min=-0.5,
                                            max=2,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m ",
                                            target=f"input_fish_tip_depth_{id}",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                        dbc.Label("Cable:"),
                        dbc.Stack(  # fish Cable
                            [
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "length unstreamline:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_cable_length_unstreamline_{id}",
                                            min=0,
                                            max=500,
                                            step=0.5,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m ",
                                            target=f"input_fish_cable_length_unstreamline_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "CX water unstreamline:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_cx_unstreamline_{id}",
                                            min=0.5,
                                            max=2,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "no unit ",
                                            target=f"input_fish_cx_unstreamline_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "length streamline:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_cable_length_streamline_{id}",
                                            min=0,
                                            max=5,
                                            step=0.5,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "unit: m ",
                                            target=f"input_fish_cable_length_streamline_{id}",
                                        ),
                                    ]
                                ),
                                dbc.Stack(
                                    [
                                        dbc.Label(
                                            "CX water streamline:",
                                            className="custom-labels",
                                        ),
                                        dbc.Input(
                                            type="number",
                                            id=f"input_fish_cx_streamline_{id}",
                                            min=0.05,
                                            max=1,
                                            step=0.01,
                                            className="custom-inputs",
                                        ),
                                        dbc.Tooltip(
                                            "no unit ",
                                            target=f"input_fish_cx_streamline_{id}",
                                        ),
                                    ]
                                ),
                            ],
                            direction="horizontal",
                        ),
                    ]
                ),
            ]
        ),
    )
    return s
