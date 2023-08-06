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
                id=f"slider-graph_size",
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
        id=f"slider-wind_speed",
        min=1,
        max=40,
        step=1,
        value=15,
        marks={i: str(i) for i in range(0, 40, 5)},
        tooltip={
            "placement": "bottom",
            "always_visible": True,
        },
    ),
    daq.BooleanSwitch(
        id="bool_orthogrid", on=True, label="Ortho grid", labelPosition="right"
    ),
    daq.BooleanSwitch(
        id="bool_backgrdimg", on=False, label="Background image", labelPosition="right"
    ),
    daq.BooleanSwitch(
        id="bool_isospeed", on=True, label="Iso speed", labelPosition="right"
    ),
    daq.BooleanSwitch(
        id="bool_isoeft", on=True, label="Iso efficiency total", labelPosition="right"
    ),
    daq.BooleanSwitch(
        id="bool_isofluid", on=True, label="Iso fluid ratio", labelPosition="right"
    ),
    dbc.Button(
        "copy parameters : Fk1 -> Fk2",
        color="secondary",
        size="sm",
        id="copy_FK0toFK1",
    ),
]

### FishKite


def create_fk_sliders(id):
    s = (
        html.Div(
            [
                html.Tr(
                    [
                        html.Td(html.H5(f"FishKite {id +1}")),
                        html.Td(
                            daq.BooleanSwitch(id=f"boolean_{id}", on=True),
                        ),
                    ]
                ),
                html.Label("rising Angle [deg]"),
                dcc.Slider(
                    id=f"slider-rising_angle_{id}",
                    min=1,
                    max=90,
                    step=1,
                    value=25,
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 90, 5)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.H6(html.B(f"Kite:")),
                html.Label("Kite Area [m2]"),
                dcc.Slider(
                    id=f"slider-kite_area_{id}",
                    min=1,
                    max=50,
                    step=1,
                    value=20,
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 55, 5)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("Kite Force Coeficient"),
                dcc.RangeSlider(
                    id=f"slider-kite_cl_{id}",
                    min=0,
                    max=1.5,
                    step=0.05,
                    value=[0.2, 0.4, 1.5],
                    updatemode="drag",
                    marks={i: str(i) for i in [0, 0.5, 1, 1.5]},
                    pushable=0,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Tr(
                    [
                        html.Td("Kite efficiency_angle [deg]:    "),
                        html.Div(id=f"label-kite_eff_LD_{id}"),
                    ]
                ),
                dcc.Slider(
                    id=f"slider-kite_efficiency_angle_{id}",
                    min=1,
                    max=45,
                    step=1,
                    value=18,
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 50, 5)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Br(),
                html.H6(html.B(f"Fish:")),
                html.Label("Fish Area [m2]"),
                dcc.Slider(
                    id=f"slider-fish_area_{id}",
                    min=0.01,
                    max=0.2,
                    step=0.01,
                    value=0.1,
                    updatemode="drag",
                    marks={i: f"{i:.2f}" for i in np.arange(0, 0.205, 0.05)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("Fish Force Coeficient"),
                dcc.RangeSlider(
                    id=f"slider-fish_cl_{id}",
                    min=0.1,
                    max=1,
                    step=0.05,
                    value=[0.1, 0.4, 1],
                    pushable=0,
                    updatemode="drag",
                    marks={i: str(i) for i in [0, 0.5, 1]},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Tr(
                    [
                        html.Td("Fish efficiency_angle [deg]:    >"),
                        html.Div(id=f"label-fish_eff_LD_{id}"),
                    ]
                ),
                dcc.Slider(
                    id=f"slider-fish_efficiency_angle_{id}",
                    min=1,
                    max=45,
                    step=1,
                    value=12,
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 50, 5)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
            ]
        ),
    )
    return s
