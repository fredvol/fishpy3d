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

# %%# Class


class Line:
    def __init__(self, pt1=None, pt2=None, center=None, length=None):
        if center is not None and length is not None:
            half_length = length / 2
            self.pt1 = np.array([center[0] - half_length, center[1]])
            self.pt2 = np.array([center[0] + half_length, center[1]])
        elif pt2 is not None:
            self.pt1 = np.array(pt1)
            self.pt2 = np.array(pt2)
        else:
            raise ValueError(
                "Invalid arguments. Provide either 'point1' and 'point2' or 'center' and 'length'."
            )

    def get_center(self):
        center = (self.point1 + self.point2) / 2
        return tuple(center)

    def move(self, dx, dy):
        self.pt1 += np.array([dx, dy])
        self.pt2 += np.array([dx, dy])

    def rotate(self, theta_degrees, center=None):
        if center is None:
            center = self.get_center()

        # Convert angle from degrees to radians
        theta_rad = np.radians(theta_degrees)

        # Create a rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)],
            ]
        )

        # Translate to the origin, rotate, and then translate back
        self.pt1 = np.dot(rotation_matrix, self.pt1 - center) + center
        self.pt2 = np.dot(rotation_matrix, self.pt2 - center) + center


# %%## Functions
def centers(y1, y2, r, x1=0, x2=0):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r**2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r**2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return (x3 + xx, y3 + yy)


def rotate_lines(line, deg=-90):
    """Rotate self.polylines the given angle about their centers."""
    theta = radians(deg)  # Convert angle from degrees to radians
    cosang, sinang = cos(theta), sin(theta)

    for pl in self.polylines:
        # Find logical center (avg x and avg y) of entire polyline
        n = len(pl.lines) * 2  # Total number of points in polyline
        cx = sum(sum(line.get_xdata()) for line in pl.lines) / n
        cy = sum(sum(line.get_ydata()) for line in pl.lines) / n

        for line in pl.lines:
            # Retrieve vertices of the line
            x1, x2 = line.get_xdata()
            y1, y2 = line.get_ydata()

            # Rotate each around whole polyline's center point
            tx1, ty1 = x1 - cx, y1 - cy
            p1x = (tx1 * cosang + ty1 * sinang) + cx
            p1y = (-tx1 * sinang + ty1 * cosang) + cy
            tx2, ty2 = x2 - cx, y2 - cy
            p2x = (tx2 * cosang + ty2 * sinang) + cx
            p2y = (-tx2 * sinang + ty2 * cosang) + cy

            # Replace vertices with updated values
            pl.set_line(line, [p1x, p2x], [p1y, p2y])


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

    dict_hover_data = {
        "vmg_x_kt": True,  # add other column, default formatting
        "vmg_y_kt": True,  # add other column, default formatting
        "apparent_watter_kt": ":.2f",  # add other column, customized formatting
        what: ":.2f",  # add other column, customized formatting
        "fk_name": True,  # add other column
        "indexG": True,  # add other column
        # # data not in dataframe, default formatting
        # "suppl_1": np.random.random(len(df)),
        # # data not in dataframe, customized formatting
        # "suppl_2": (":.3f", np.random.random(len(df))),
    }
    # symbol management
    fix_list = ["circle", "circle-open", "x", "hash", "hexagrame"]
    custom_symbol_sequence = []
    if symbol is not None:
        if len(dfs[symbol].unique()) <= len(fix_list):
            custom_symbol_sequence = fix_list

    fig = px.scatter(
        dfs,
        x="vmg_x_kt",
        y="vmg_y_kt",
        color=what,
        symbol=symbol,
        hover_data=dict_hover_data,
        symbol_sequence=custom_symbol_sequence,
    )

    # UPdate layout
    fig.update_layout(
        title=go.layout.Title(
            text=f"Rising angle: [{target_rising_angle_low},{target_rising_angle_upper}] , TrueWind = {target_wind} kt"
        ),
        autosize=True,
        plot_bgcolor="rgba(240,240,240,0.7)",
        xaxis=dict(showgrid=False, visible=False),
        height=height_size,
        # width=height_size*0.6,
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
            hoverinfo="skip",
        )
    )

    fig.update_yaxes(  # make suare ratio
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

    fig.update_xaxes(range=[-5, 50], constrain="domain")
    fig.update_yaxes(range=[-40, 20], constrain="domain")

    fig.update_layout(coloraxis_colorbar_x=0.9)
    fig.update_layout(legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.85))
    fig.update_layout(clickmode="event+select")
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


def plot_side_view(row, fk1):
    span_fish = fk1.fish.projected_span()
    span_kite = fk1.kite.projected_span()

    center_fish = (0, -row["fish_center_depth"])
    fish_line = Line(center=center_fish, length=span_fish)
    fish_line.rotate(theta_degrees=90 + row["rising_angle"], center=center_fish)

    center_kite = (row["y_kite"], row["z_kite"])
    kite_line = Line(center=center_kite, length=span_kite)
    kite_line.rotate(theta_degrees=90 + row["kite_roll_angle"], center=center_kite)

    fig = go.Figure()
    fig.update_layout(
        title=go.layout.Title(text=f"Side view"),
        # autosize=True,
        plot_bgcolor="rgba(240,240,240,0.7)",
        height=320,
        # width=height_size * 0.8,
        xaxis_range=[-1, 45],
        legend=dict(
            # y=1,
            # x=0.1,
            groupclick="toggleitem",
        ),
    )

    # water
    fig.add_trace(
        add_line(
            (-1, 0),
            (row["y_kite"] + 1, 0),
            m_name="water",
            group_name="water",
            extra_dict=dict(width=2, color="blue"),
        )
    )

    # fishcable
    fig.add_trace(
        add_line(
            center_fish,
            (row["y_pilot"], row["z_pilot"]),
            m_name="fish_cable",
            group_name="fish_cable",
            extra_dict=dict(width=3, color="red"),
        )
    )

    # fish
    fig.add_trace(
        add_line(
            fish_line.pt1,
            fish_line.pt2,
            m_name="fish",
            group_name="fish",
            extra_dict=dict(width=3, color="black"),
        )
    )

    # Kitecable
    fig.add_trace(
        add_line(
            (row["y_pilot"], row["z_pilot"]),
            (row["y_kite"], row["z_kite"]),
            m_name="Kite_cable",
            group_name="Kite_cable",
            extra_dict=dict(width=3, color="green"),
        )
    )

    # kite
    fig.add_trace(
        add_line(
            kite_line.pt1,
            kite_line.pt2,
            m_name="kite",
            group_name="kite",
            extra_dict=dict(width=3, color="black"),
        )
    )

    # #add pilot
    # fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     x0=1, y0=1, x1=3, y1=3,
    #     line_color="LightSeaGreen",
    # )

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
    fk2 = FishKite.from_json(fk2_file)

    df = fk1.create_df()

    # addmising col as is not comming from a project
    df["fk_name"] = "fk1"
    df["indexG"] = df.index

    fig = plot_3d_cases_risingangle(
        df,
        target_rising_angle_low=20,
        target_rising_angle_upper=30,
        what="z_kite",
        symbol="failure",
        target_wind=30,
        draw_ortho_grid=True,
        draw_iso_speed=True,
        draw_iso_eft=True,
        draw_iso_fluid=True,
    )
    fig.show()

# %%
