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


data_background = {
    "iso_speed": {
        "step": range(10, 50, 10),
        "color": "#00FFFF",
        "opacity": 0.4,
    },
    "iso_eft": {
        "step": [90, 80, 70, 60, 50, 40, 30, 20, 15, 12, 10],
        "extra_step": [15, 12],
        "color": "black",
        "opacity": 0.2,
    },
    "iso_fluid": {
        # "step": [10.0, 5.0, 3.34, 2.5, 2.0, 1.67, 1.43, 1.25, 1.12],
        "step": [10.0, 5.0, 3.34, 2.5, 2.0, 1.67, 1.43, 1.25, 1.12],
        "extra_step": [],  # [0.95, 0.98],
        "color": "green",
        "opacity": 0.3,
    },
}


### Functions
def centers(y1, y2, r, x1=0, x2=0):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r**2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r**2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return (x3 + xx, y3 + yy)


def add_line(
    pt1,
    pt2,
    m_name,
    group_name=None,
    legendgrouptitle_text=None,
    extra_dict=None,
):
    base_line_dict = dict(
        color=None,
    )

    if extra_dict is not None:
        base_line_dict.update(extra_dict)

    return go.Scatter(
        mode="lines",
        x=[pt1[0], pt2[0]],
        y=[pt1[1], pt2[1]],
        legendgroup=group_name,
        legendgrouptitle_text=legendgrouptitle_text,
        name=m_name,
        line=base_line_dict,
    )


def ellipse_arc(
    x_center=0,
    y_center=0,
    a=1,
    b=1,
    start_angle=0,
    end_angle=2 * np.pi,
    N=100,
    closed=False,
):
    t = np.linspace(start_angle, end_angle, N)
    x = x_center + a * np.cos(t)
    y = y_center + b * np.sin(t)
    path = f"M {x[0]}, {y[0]}"
    for k in range(1, len(t)):
        path += f"L{x[k]}, {y[k]}"
    if closed:
        path += " Z"
    return path


def create_iso_eft(dict_param, wind_speed, position_angle_label=30):
    # wind_speed = 100  # assuming wins speed = 100%
    isoeft = []
    isoeft_label = []

    for s in dict_param["step"]:
        s_rad = np.radians(s)
        r = (wind_speed / 2) / (np.sin(s_rad))
        center_x, center_y = centers(0, -wind_speed, r)
        angle = np.pi - (s_rad)

        x_label = r * np.cos(np.radians(position_angle_label)) + center_x
        y_label = r * np.sin(np.radians(position_angle_label)) + center_y
        label_s = {
            "x": x_label,
            "y": y_label,
            "text": f"Et={s}°",
            "angle": position_angle_label,
            "color": dict_param["color"],
            "opacity": dict_param["opacity"] + 0.1,
        }

        isoeft_label.append(label_s)
        isoeft.append(
            dict(
                type="path",
                path=ellipse_arc(
                    x_center=center_x,
                    y_center=center_y,
                    a=r,
                    b=r,
                    start_angle=-angle,
                    end_angle=angle,
                    N=60,
                ),
                # line_color=dict_param["color"],
                opacity=dict_param["opacity"],
                layer="below",
                line=dict(
                    color=dict_param["color"],
                    width=1,
                    dash="dot" if s in dict_param["extra_step"] else None,
                ),
                # dash="dashdot" if s in dict_param["extra_step"] else None,
            )
        )
    df_label = pd.DataFrame(isoeft_label)
    return isoeft, df_label


def create_iso_speed(dict_param, position_angle_label=30):
    isospeed = []
    isospeed_label = []
    for s in dict_param["step"]:
        x_label = s * np.cos(np.radians(position_angle_label))
        y_label = s * np.sin(np.radians(position_angle_label))
        label_s = {
            "x": x_label,
            "y": y_label,
            "text": f"{s}kt wind",
            "angle": 90 - position_angle_label,
            "color": dict_param["color"],
            "opacity": dict_param["opacity"] + 0.1,
        }

        isospeed_label.append(label_s)

        isospeed.append(
            dict(
                type="path",
                path=ellipse_arc(
                    a=s, b=s, start_angle=-np.pi / 2, end_angle=np.pi / 2, N=60
                ),
                line=dict(
                    color=dict_param["color"],
                    width=1,
                ),
                opacity=dict_param["opacity"],
                layer="below",
            )
        )
    df_label = pd.DataFrame(isospeed_label)
    return isospeed, df_label


def create_iso_fluid(dict_param, wind_speed, position_angle_label=10):
    # wind_speed = 100  # assuming wins speed = 100%
    isofluid = []
    isofluid_label = []

    # add straight line
    isofluid.append(
        dict(
            type="line",
            x0=1,
            y0=-wind_speed / 2,
            x1=300,
            y1=-wind_speed / 2,
            opacity=dict_param["opacity"],
            layer="below",
            line=dict(
                color=dict_param["color"],
                width=1,
            ),
        )
    )
    label_straight = {
        "x": 535,
        "y": -44,
        "text": f"ratio:1",
        "angle": 0,
        "color": dict_param["color"],
        "opacity": dict_param["opacity"],
    }

    isofluid_label.append(label_straight)

    # add others

    for k in dict_param["step"] + dict_param["extra_step"]:
        r = (wind_speed) * k / (k**2 - 1)
        center_x = 0
        center_y = -wind_speed - (wind_speed / (k**2 - 1))

        if (center_y - r - 5) > -300:  # TODO to sort the label position
            x_label = 5
            y_label = center_y - r - 5
        else:
            dict_x_position = {1.12: 420, 1.25: 240, 1.43: 105}

            x_label = dict_x_position[k]
            y_label = -300

        label_f = {
            "x": x_label,
            "y": y_label,
            "text": f"ratio:{round(k,2)}",
            "angle": 0,
            "color": dict_param["color"],
            "opacity": dict_param["opacity"],
        }

        isofluid_label.append(label_f)

        isofluid.append(
            dict(
                type="path",
                path=ellipse_arc(
                    x_center=center_x,
                    y_center=center_y,
                    a=r,
                    b=r,
                    start_angle=np.pi / 2,
                    end_angle=-np.pi / 2,
                    N=60,
                ),
                line=dict(
                    color=dict_param["color"],
                    width=1,
                ),
                opacity=dict_param["opacity"],
                layer="below",
            )
        )

        # Miror

        label_fs = {
            "x": x_label,
            "y": -y_label - wind_speed,
            "text": f"ratio: {round(1/k,2)}",
            "angle": position_angle_label,
            "color": dict_param["color"],
            "opacity": dict_param["opacity"],
        }

        isofluid_label.append(label_fs)

        isofluid.append(
            dict(
                type="path",
                path=ellipse_arc(
                    x_center=center_x,
                    y_center=-center_y - wind_speed,
                    a=r,
                    b=r,
                    start_angle=np.pi / 2,
                    end_angle=-np.pi / 2,
                    N=60,
                ),
                line=dict(
                    color=dict_param["color"],
                    width=1,
                ),
                opacity=dict_param["opacity"],
                layer="below",
            )
        )
    df_label = pd.DataFrame(isofluid_label)
    return isofluid, df_label


################# MAIN PLOT FUNCTION ############


def plot_3d_cases_risingangle(
    df,
    target_rising_angle_low=20,
    target_rising_angle_upper=40,
    target_wind=30,
    what="extra_angle",
    symbol=None,
    height_size=850,
    draw_ortho_grid=True,
    draw_iso_speed=True,
    draw_iso_eft=True,
    draw_iso_fluid=True,
):
    shape_list = []

    # add shapes
    if draw_iso_speed:
        shapes_speed, labels_speed = create_iso_speed(
            data_background["iso_speed"], position_angle_label=10
        )
        shape_list.extend(shapes_speed)

    if draw_iso_eft:
        shapes_eft, labels_eft = create_iso_eft(
            data_background["iso_eft"], target_wind, position_angle_label=-45
        )
        shape_list.extend(shapes_eft)

    if draw_iso_fluid:
        shapes_isofluid, labels_isofluid = create_iso_fluid(
            data_background["iso_fluid"], target_wind, position_angle_label=15
        )
        shape_list.extend(shapes_isofluid)

    # fig = go.Figure(
    #     layout=go.Layout(
    #         title=go.layout.Title(
    #             text=f"Rising angle: [{target_rising_angle_low},{target_rising_angle_upper}] , TrueWind = {target_wind} kt"
    #         ),
    #         autosize=True,
    #         plot_bgcolor="rgba(240,240,240,0.7)",
    #         xaxis=dict(showgrid=False, visible=False),
    #         yaxis=dict(showgrid=False, visible=False),
    #         legend=dict(
    #             y=1,
    #             x=0.8,
    #             groupclick="toggleitem",
    #         ),
    #         margin=dict(l=50, r=50, b=5, t=30, pad=3),
    #     ),
    # )

    dfs = df[
        (df["rising_angle"].between(target_rising_angle_low, target_rising_angle_upper))
        & (df["true_wind_calculated_kt_rounded"] == target_wind)
    ]

    list_hover_data = [
        "apparent_watter_kt",
        what,
        "indexG",  # index G should remain the last  for the side table data
    ]

    fig = px.scatter(
        dfs,
        x="vmg_x_kt",
        y="vmg_y_kt",
        color=what,
        symbol=symbol,
        hover_data=list_hover_data,
    )

    # UPdate layout
    fig.update_layout(
        title=go.layout.Title(
            text=f"Rising angle: [{target_rising_angle_low},{target_rising_angle_upper}] , TrueWind = {target_wind} kt"
        ),
        # autosize=True,
        plot_bgcolor="rgba(240,240,240,0.7)",
        xaxis=dict(showgrid=False, visible=False),
        height=height_size,
        # width=height_size * 0.8,
        yaxis=dict(showgrid=False, visible=False),
        xaxis_range=[-1, 45],
        legend=dict(
            y=1,
            x=0.8,
            groupclick="toggleitem",
        ),
        margin=dict(l=50, r=50, b=5, t=30, pad=3),
        shapes=shape_list,
    )

    # add wind
    # Wind
    fig.add_trace(
        add_line(
            (0, 0),
            (0, -target_wind),
            m_name="wind",
            group_name="Wind",
            extra_dict=dict(width=3, color="red"),
        )
    )

    # add simplify

    dfsimplify = dfs[dfs["simplify"]]

    fig.add_trace(
        go.Scatter(
            x=dfsimplify["vmg_x_kt"],
            y=dfsimplify["vmg_y_kt"],
            mode="markers",
            name="simplify",
            marker=dict(
                size=12,
                color="black",
            ),
        )
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    if draw_ortho_grid:
        fig.update_layout(
            xaxis=dict(showgrid=True, visible=True),
            yaxis=dict(showgrid=True, visible=True),
        )

    if draw_iso_speed:
        for i, row in labels_speed.iterrows():
            fig.add_annotation(
                go.layout.Annotation(
                    x=row["x"],
                    y=row["y"],
                    text=row["text"],
                    hovertext=row["text"],
                    # textposition="bottom center"
                    textangle=row["angle"],
                    # arrowsize=0.3,
                    showarrow=False,
                    font=dict(color=row["color"]),
                    # for color use font dict
                    opacity=row["opacity"],
                    xshift=10,
                    yshift=5,
                )
            )
    if draw_iso_eft:
        for i, row in labels_eft.iterrows():
            fig.add_annotation(
                go.layout.Annotation(
                    x=row["x"],
                    y=row["y"],
                    text=row["text"],
                    hovertext=row["text"],
                    # textposition="bottom center"
                    textangle=row["angle"],
                    arrowsize=0.3,
                    font=dict(color=row["color"]),
                    opacity=row["opacity"],
                )
            )
    if draw_iso_fluid:
        for i, row in labels_isofluid.iterrows():
            fig.add_annotation(
                go.layout.Annotation(
                    x=row["x"],
                    y=row["y"],
                    text=row["text"],
                    hovertext=row["text"],
                    textangle=0,
                    showarrow=False,
                    font=dict(color=row["color"], size=8),
                    opacity=row["opacity"],
                )
            )

    # fig.update_layout(
    #     xaxis_range=[0, 60],
    #     yaxis_range=[-target_wind-20, target_wind+10],

    # )

    fig.update_xaxes(range=[-1, 45])
    fig.update_yaxes(range=[-35, 30])

    fig.update_layout(coloraxis_colorbar_x=-0.005)
    return fig


def plot_3d_cases(df, target_wind=30, what="rising_angle", height_size=850):
    dfs = df[df["true_wind_calculated_kt_rounded"] == target_wind]
    fig = go.Figure(
        data=go.Scattergl(
            x=dfs["vmg_x_kt"],
            y=dfs["vmg_y_kt"],
            mode="markers",
            marker=dict(color=dfs[what], colorscale="Viridis", line_width=1),
            hovertext=dfs[what],
        )
    )

    fig.update_layout(
        title=go.layout.Title(text=f" All polar points "),
        autosize=True,
        plot_bgcolor="rgba(240,240,240,0.7)",
        height=height_size,
        width=height_size * 1.1,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# %% Initial set up

if __name__ == "__main__":
    from model_3d import Deflector, FishKite, Pilot

    data_folder = os.path.join(os.path.dirname(__file__), "data")  # we go up 2 folders

    fk1_file = os.path.join(data_folder, "saved_fk1.json")
    fk2_file = os.path.join(data_folder, "saved_fk2.json")

    fk1 = FishKite.from_json(fk1_file)
    fk2 = FishKite.from_json(fk2_file, classes=FishKite)

    # # %%
    df = fk1.create_df()

    fig = plot_3d_cases_risingangle(
        df,
        target_rising_angle_low=20,
        target_rising_angle_upper=30,
        symbol="failure",
        draw_ortho_grid=True,
        draw_iso_speed=True,
        draw_iso_eft=True,
        draw_iso_fluid=True,
    )
    fig.show()
# %%
