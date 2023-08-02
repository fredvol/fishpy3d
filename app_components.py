from dash import dcc
from dash import html
import dash_daq as daq
import numpy as np

### Operating things
operating_slider_components = [
    html.Label("Wind speed (knots)"),
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
]

### FishKite


def create_fk_sliders(id):
    s = (
        html.Div(
            [
                html.H5(f"FishKite {id +1}"),
                daq.BooleanSwitch(id=f"boolean_{id}", on=True),
                # list_of_controls
                html.Label("rising Angle (deg)"),
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
                html.Label("Area"),
                dcc.Slider(
                    id=f"slider-kite_area_{id}",
                    min=1,
                    max=40,
                    step=1,
                    value=20,
                    updatemode="drag",
                    marks={i: str(i) for i in range(0, 40, 5)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("Cl"),
                dcc.RangeSlider(
                    id=f"slider-kite_cl_{id}",
                    min=0,
                    max=1.5,
                    step=0.1,
                    value=[0.2, 0.4, 1.5],
                    updatemode="drag",
                    marks={i: str(i) for i in [0, 0.5, 1, 1.5]},
                    pushable=0,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("efficiency_angle"),
                dcc.Slider(
                    id=f"slider-kite_efficiency_angle_{id}",
                    min=0,
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
                html.Label("Area"),
                dcc.Slider(
                    id=f"slider-fish_area_{id}",
                    min=0.01,
                    max=1.2,
                    step=0.01,
                    value=0.1,
                    updatemode="drag",
                    marks={i: f"{i:.2f}" for i in np.arange(0, 1.4, 0.2)},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("Cl"),
                dcc.RangeSlider(
                    id=f"slider-fish_cl_{id}",
                    min=0.1,
                    max=1,
                    step=0.1,
                    value=[0.1, 0.4, 1],
                    pushable=0,
                    updatemode="drag",
                    marks={i: str(i) for i in [0, 0.5, 1]},
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },
                ),
                html.Label("efficiency_angle"),
                dcc.Slider(
                    id=f"slider-fish_efficiency_angle_{id}",
                    min=0,
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
