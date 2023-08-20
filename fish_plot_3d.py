# %% FISH Kite plot functions
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# %% plot data
COLOR_palette = px.colors.qualitative.Plotly
COLOR_palette[2] = "#28a745"

if "#EF553B" in COLOR_palette:
    COLOR_palette.remove("#EF553B")  # "#AB63FA"


# bckgrd_imge_dim = {
#     "width": 700,
#     "height": 636,
#     "zero_x": 31,
#     "zero_y": 265,
#     "source": "assets\\polar_background_small.png",
# }


def plot_3d_cases_risingangle(
    list_of_cases,
    height_size=900,
    target_rising_angle=20,
    target_wind=30,
):
    df_list = []

    for fk in list_of_cases:
        dfi = fk.create_df()
        dfi["fk_name"] = fk.name
        df_list.append(dfi)

    df = pd.concat(df_list, ignore_index=True)

    title_cases = ", ".join([fk.name for fk in list_of_cases])

    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=f"Polar for: {title_cases}"),
            autosize=True,
            plot_bgcolor="rgba(240,240,240,0.7)",
            height=height_size,
            width=height_size * 1.1,
            xaxis_range=[-15, 550],
            yaxis_range=[-300, 240],
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            legend=dict(
                y=1,
                x=0.8,
                groupclick="toggleitem",
            ),
            margin=dict(l=50, r=50, b=5, t=30, pad=3),
        ),
    )

    dfs = df[
        (df["rising_angle"] == target_rising_angle)
        & (df["true_wind_calculated_kt_rounded"] == target_wind)
    ]
    fig = px.scatter(
        dfs,
        x="vmg_x_kt",
        y="vmg_y_kt",
        color="extra_angle",
        title=f"Polar pts for rising angle:{target_rising_angle} and TW= {target_wind} kt",
    )
    # fig.update_traces(marker=dict(color="red"))

    dfsimplify = dfs[dfs["simplify"] == 1]

    fig.add_trace(
        go.Scatter(
            x=dfsimplify["vmg_x_kt"],
            y=dfsimplify["vmg_y_kt"],
            mode="markers",
            marker=dict(
                size=10,
                # I want the color to be green if
                # lower_limit ≤ y ≤ upper_limit
                # else red
                color="red",
            ),
        )
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def plot_3d_cases(
    df,
):
    title_cases = "3D plot "

    fig = go.Figure(
        data=go.Scattergl(
            x=df["vmg_x_kt"],
            y=df["vmg_y_kt"],
            mode="markers",
            marker=dict(color=df["rising_angle"], colorscale="Viridis", line_width=1),
        )
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# %%
