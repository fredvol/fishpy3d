# %% FISH Kite plot functions
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# %% plot data
COLOR_palette = px.colors.qualitative.Plotly

# bckgrd_imge_dim = {
#     "width": 700,
#     "height": 636,
#     "zero_x": 31,
#     "zero_y": 265,
#     "source": "assets\\polar_background_small.png",
# }

data_background = {
    "iso_speed": {
        "step": range(100, 600, 100),
        "color": "PaleTurquoise",
        "opacity": 0.9,
    },
    "iso_eft": {
        "step": [90, 80, 70, 60, 50, 40, 30, 20, 15, 12, 10],
        "color": "black",
        "opacity": 0.08,
    },
}

bckgrd_imge_dim = {
    "width": 867,
    "height": 800,
    "zero_x": 68,
    "zero_y": 333,
    "source": "assets\\polar_background.jpeg",
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
    pt1, pt2, m_name, group_name=None, legendgrouptitle_text=None, extra_dict=None
):
    based_dict = dict(
        color=None,
    )

    if extra_dict is not None:
        based_dict.update(extra_dict)

    return go.Scatter(
        mode="lines",
        x=[pt1[0], pt2[0]],
        y=[pt1[1], pt2[1]],
        legendgroup=group_name,
        legendgrouptitle_text=legendgrouptitle_text,
        name=m_name,
        line=based_dict,
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


def create_iso_eft(dict_param, position_angle_label=30):
    wind_speed = 100  # assuming wins speed = 100
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
                line_color=dict_param["color"],
                opacity=dict_param["opacity"],
                layer="below",
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
            "text": f"{s}% wind",
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
                line_color=dict_param["color"],
                opacity=dict_param["opacity"],
                layer="below",
            )
        )
    df_label = pd.DataFrame(isospeed_label)
    return isospeed, df_label


def plot_cases(
    list_of_cases,
    draw_ortho_grid=True,
    draw_iso_speed=True,
    draw_iso_eft=True,
    add_background_image=False,
    height_size=800,
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
            data_background["iso_eft"], position_angle_label=-45
        )
        shape_list.extend(shapes_eft)

    title_cases = ", ".join([fk.name for fk in list_of_cases])

    graph_size = bckgrd_imge_dim["height"] if add_background_image else height_size
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=f"Polar for: {title_cases}"),
            autosize=True,
            plot_bgcolor="rgba(240,240,240,0.7)",
            height=graph_size,
            width=graph_size * 1.1,
            xaxis_range=[-5, 550],
            yaxis_range=[-300, 240],
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            legend=dict(
                y=1,
                x=0.8,
                groupclick="toggleitem",
            ),
            margin=dict(l=50, r=50, b=5, t=30, pad=3),
            shapes=shape_list,
        ),
    )
    for i, fki in enumerate(list_of_cases):
        color = COLOR_palette[i]
        fki.add_plot_elements(fig, m_color=color, add_legend_name=True)

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )  # to keep square ratio

    if draw_ortho_grid:
        fig.update_layout(
            xaxis=dict(showgrid=True, visible=True),
            yaxis=dict(showgrid=True, visible=True),
        )

    if add_background_image:
        fig.update_layout(
            width=bckgrd_imge_dim["width"],
            height=bckgrd_imge_dim["height"],
        )  # to keep square ratio
        fig.add_layout_image(
            dict(
                source=bckgrd_imge_dim["source"],
                xref="x",
                yref="y",
                x=-bckgrd_imge_dim["zero_x"] * 0.75,
                y=bckgrd_imge_dim["zero_y"] * 0.75,
                sizex=bckgrd_imge_dim["width"] * 0.75,
                sizey=bckgrd_imge_dim["height"] * 0.75,  # TODO why 0.75?
                sizing="stretch",
                xanchor="left",
                yanchor="top",
                opacity=0.4,
                layer="below",
            )
        )

    # add labels

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

    return fig
