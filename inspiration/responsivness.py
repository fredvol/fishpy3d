import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np

app = dash.Dash(__name__)

# plot a circle
t = np.linspace(
    0, 2 * np.pi, 10000
)  # intentionally use many points to exaggerate the issue
x = np.cos(t)
y = np.sin(t)
z = np.zeros_like(t)
marker_size = 4
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines"))
fig.add_trace(
    go.Scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        marker=go.scatter3d.Marker(size=marker_size),
        line=dict(width=3.0),
        showlegend=False,
    )
)

fig.update_layout(
    uirevision="constant",
    autosize=False,
    width=900,
    height=900,
    scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1.0]),
        aspectratio=dict(x=2, y=2, z=2),
    ),
)

app.layout = html.Div(
    children=[
        dcc.Graph(id="example-graph", figure=fig),
        html.Div(
            [
                dcc.Slider(
                    id="slider-phi",
                    min=0.0,
                    max=360.0,
                    step=1.0,
                    value=0.0,
                    marks={0: "0", 180: "180", 360: "360"},
                    updatemode="drag",
                ),
            ],
            style=dict(width="50%"),
        ),
        html.Div(children="", id="output-box"),
    ]
)

app.clientside_callback(
    """
    function(phi) {
        // tuple is (dict of new data, target trace index, number of points to keep)
        return [{x: [[0, Math.cos(phi/180*Math.PI)]], y:[[0, Math.sin(phi/180*Math.PI)]]}, [1], 2]
    }
    """,
    Output("example-graph", "extendData"),
    [Input("slider-phi", "value")],
)

if __name__ == "__main__":
    app.run_server(debug=True)
