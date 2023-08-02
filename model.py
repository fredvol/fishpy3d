# %% FISH Kite model
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


# %% constants
RHO_AIR = 1.29  # kg/m3
RHO_WATER = 1025  # kg/m3

# %% plot data
COLOR_palette = px.colors.qualitative.Plotly

bckgrd_imge_dim = {
    "width": 700,
    "height": 636,
    "zero_x": 31,
    "zero_y": 265,
    "source": "assets\\polar_background_small.png",
}

data_background = {
    "iso_speed": {
        "step": range(100, 600, 100),
        "color": "PaleTurquoise",
        "opacity": 0.6,
    },
    "iso_eft": {
        "step": range(100, 600, 100),
        "color": "black",
        "opacity": 0.1,
    },
}

# bckgrd_imge_dim = {
#     "width": 1083,
#     "height": 1000,
#     "zero_x": 82,
#     "zero_y": 416,
#     "source": "assets\\polar_background.jpeg",
# }

# %%  Functions


def centers(y1, y2, r, x1=0, x2=0):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r**2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r**2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return (x3 + xx, y3 + yy)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi, start=90):
    phirad = np.radians(start - phi)
    x = rho * np.cos(phirad)
    y = rho * np.sin(phirad)
    return (x, y)


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


def create_iso_eft(dict_param):
    isoeft = []
    for s in dict_param["step"]:
        center_x, center_y = centers(0, -100, s)
        angle = np.pi - np.tan(50 / s)

        isoeft.append(
            dict(
                type="path",
                path=ellipse_arc(
                    x_center=center_x,
                    y_center=center_y,
                    a=s,
                    b=s,
                    start_angle=-angle,
                    end_angle=angle,
                    N=60,
                ),
                line_color=dict_param["color"],
                opacity=dict_param["opacity"],
                layer="below",
            )
        )
    return isoeft


def create_iso_speed(dict_param):
    isospeed = []
    for s in dict_param["step"]:
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
    return isospeed


def plot_cases(
    list_of_cases,
    draw_ortho_grid=True,
    draw_iso_speed=True,
    draw_iso_eft=True,
    add_background_image=False,
):
    shape_list = []

    if draw_iso_speed:
        shape_list.extend(create_iso_speed(data_background["iso_speed"]))

    if draw_iso_eft:
        shape_list.extend(create_iso_eft(data_background["iso_eft"]))

    title_cases = ", ".join([fk.name for fk in list_of_cases])
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=f"Polar for: {title_cases}"),
            autosize=False,
            height=bckgrd_imge_dim["height"],
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
        fig.add_layout_image(
            dict(
                source=bckgrd_imge_dim["source"],
                xref="x",
                yref="y",
                x=-bckgrd_imge_dim["zero_x"],
                y=bckgrd_imge_dim["zero_y"],
                sizex=bckgrd_imge_dim["width"] - bckgrd_imge_dim["zero_x"],
                sizey=bckgrd_imge_dim["height"] - 10,  # TODO why -10?
                sizing="stretch",
                xanchor="left",
                yanchor="top",
                opacity=0.4,
                layer="below",
            )
        )
    return fig


# %% Class


class Deflector:
    # init method or constructor
    def __init__(
        self,
        name: str,
        cl: float,
        cl_range: tuple,
        area: float,
        efficiency_angle: float,
    ):
        self.name = name
        self.cl = cl
        self.area = area  # m2
        self.efficiency_angle = efficiency_angle  # deg

        self.cl_range = {"min": cl_range[0], "max": cl_range[1]}

    def __repr__(self):
        return f"{self.name} Cl:{self.cl} efficiency_angle:{self.efficiency_angle} Area:{self.area} "

    # Sample Method
    def glide_ratio(self) -> float:
        return 1 / np.tan(np.radians(self.efficiency_angle))

    def projected_efficiency_angle(self, m_raising_angle) -> float:
        return np.degrees(
            np.arctan(1 / (np.cos(np.radians(m_raising_angle)) * self.glide_ratio()))
        )


class FishKite:
    def __init__(
        self,
        name: str,
        wind_speed: float,
        rising_angle: float,
        fish: Deflector,
        kite: Deflector,
    ):
        self.name = name
        self.wind_speed = wind_speed  # kt
        self.rising_angle = rising_angle  # deg
        self.fish = fish
        self.kite = kite

    def __repr__(self):
        return f"FishKite({self.name}): wind_speed:{self.wind_speed} rising_angle:{self.rising_angle}  \n Kite:{self.kite}  \n Fish:{self.fish}"

    def projected_efficiency_angle(self, what: str) -> float:
        if what == "kite":
            return self.kite.projected_efficiency_angle(self.rising_angle)
        elif what == "fish":
            return self.fish.projected_efficiency_angle(self.rising_angle)
        else:
            print(f" {what} is unkown , waiting for 'fish' or 'kite' ")

    def total_efficiency(self):
        return self.kite.projected_efficiency_angle(
            self.rising_angle
        ) + self.fish.projected_efficiency_angle(self.rising_angle)

    def fluid_velocity_ratio(self):
        current_ratio = (
            RHO_AIR
            * self.kite.area
            * self.kite.cl
            / (RHO_WATER * self.fish.area * self.fish.cl)
        ) ** 0.5
        return current_ratio

    def fluid_velocity_ratio_range(self):
        min_ratio = (
            RHO_AIR
            * self.kite.area
            * self.kite.cl_range["min"]
            / (RHO_WATER * self.fish.area * self.fish.cl_range["max"])
        ) ** 0.5
        max_ratio = (
            RHO_AIR
            * self.kite.area
            * self.kite.cl_range["max"]
            / (RHO_WATER * self.fish.area * self.fish.cl_range["min"])
        ) ** 0.5

        return {"max": max_ratio, "min": min_ratio}

    def true_wind_angle(self, velocity_ratio):
        # TODO to clean
        value = np.degrees(
            np.arctan(
                np.sin(np.radians(self.total_efficiency()))
                / (velocity_ratio - np.cos(np.radians(self.total_efficiency())))
            )
        )
        if value > 0:
            return 180 - value
        else:
            return 180 - (180 + value)

    def apparent_wind(self, velocity_ratio):
        apparent_wind_kt = (
            self.wind_speed
            * np.sin(np.radians(180 - self.true_wind_angle(velocity_ratio)))
            / np.sin(np.radians(self.total_efficiency()))
        )
        return apparent_wind_kt

    def apparent_watter(self, velocity_ratio):
        apparent_water_kt = self.fluid_velocity_ratio() * self.apparent_wind(
            velocity_ratio
        )
        return apparent_water_kt

    def compute_polar(self, nb_value=30):
        velocity_max_min = self.fluid_velocity_ratio_range()
        velocity_range = np.linspace(
            velocity_max_min["min"], velocity_max_min["max"], nb_value
        )

        list_result = []
        for velocity_ratio in velocity_range:
            dict_i = {
                "velocity_ratio": velocity_ratio,
                "true_wind_angle": self.true_wind_angle(velocity_ratio),
                "apparent_wind_kt": self.apparent_wind(velocity_ratio),
            }
            list_result.append(dict_i)

        df_polar = pd.DataFrame(list_result)

        df_polar["apparent_wind_pct"] = (
            df_polar["apparent_wind_kt"] / self.wind_speed * 100
        )
        df_polar["apparent_watter_kt"] = (
            df_polar["velocity_ratio"] * df_polar["apparent_wind_kt"]
        )
        df_polar["apparent_watter_pct"] = (
            df_polar["apparent_watter_kt"] / self.wind_speed * 100
        )
        df_polar["x_watter_pct"] = df_polar["apparent_watter_pct"] * np.sin(
            np.radians(df_polar["true_wind_angle"])
        )
        df_polar["y_watter_pct"] = df_polar["apparent_watter_pct"] * np.cos(
            np.radians(df_polar["true_wind_angle"])
        )

        return df_polar

    def data_to_plot_polar(self):
        vr = self.fluid_velocity_ratio()
        current_apparent_watter_pct = self.apparent_watter(vr) / self.wind_speed * 100
        current_true_wind_angle = self.true_wind_angle(vr)

        anchor = [0, 0]
        wind = [0, -100]
        op_point = pol2cart(current_apparent_watter_pct, current_true_wind_angle)
        polar_pts = self.compute_polar()

        return {
            "anchor": anchor,
            "wind": wind,
            "op_point": op_point,
            "polar_pts": polar_pts,
        }

    def plot(self, draw_ortho_grid=True, add_background_image=False):
        fig = plot_cases(self, [self], draw_ortho_grid, add_background_image)
        return fig

    def add_plot_elements(self, fig, m_color=None, add_legend_name=False):
        data_plot = self.data_to_plot_polar()

        legend_name = ""
        if add_legend_name:
            legend_name = "_" + self.name

        # Wind
        fig.add_trace(
            add_line(
                data_plot["anchor"],
                data_plot["wind"],
                m_name="wind",
                group_name="Wind",
                extra_dict=dict(width=4, color="red"),
            )
        )
        fig.add_trace(
            (
                go.Scatter(
                    x=data_plot["polar_pts"]["x_watter_pct"],
                    y=data_plot["polar_pts"]["y_watter_pct"],
                    mode="lines",
                    legendgrouptitle_text=self.name,
                    name=f"Polar{legend_name}",
                    line_color=m_color,
                )
            )
        )

        # trajectory
        fig.add_trace(
            add_line(
                data_plot["anchor"],
                data_plot["op_point"],
                m_name=f"trajectory{legend_name} ",
                group_name=self.name,
                extra_dict=dict(dash="dash", color=m_color),
            )
        )

        # Apparent_wind_vector
        fig.add_trace(
            add_line(
                data_plot["wind"],
                data_plot["op_point"],
                m_name=f"Apparent_wind_vector{legend_name}",
                group_name=self.name,
                extra_dict=dict(dash="dot", color=m_color),
            )
        )

        return fig


class Project:
    def __init__(self, lst_fishkite=[], name="Project1"):
        self.name = name
        self.lst_fishkite = lst_fishkite

    def __str__(self):
        return f"{self.name}"

    def detail(self):
        detail_str = f"Project conatains {len(self.lst_fishkite)} FiskKite(s):"
        for i in self.lst_fishkite:
            detail_str += "\n-\n"
            detail_str += str(i)
        return detail_str

    def plot(self, draw_ortho_grid=True, add_background_image=False):
        fig = plot_cases(self.lst_fishkite, draw_ortho_grid, add_background_image)
        return fig


# %% Parameter
if __name__ == "__main__":
    wind_speed_i = 15  # kt
    rising_angle_1 = 33  # deg
    rising_angle_2 = 20  # deg

    d_kite1 = Deflector(
        "kite1", cl=0.4, cl_range=(0.4, 0.9), area=24, efficiency_angle=12
    )
    d_fish1 = Deflector(
        "fish1", cl=0.2, cl_range=(0.2, 0.4), area=0.1, efficiency_angle=14
    )

    d_kite2 = Deflector(
        "kite2", cl=0.6, cl_range=(0.4, 0.9), area=25, efficiency_angle=4
    )
    d_fish2 = Deflector(
        "fish2", cl=0.4, cl_range=(0.2, 0.4), area=0.07, efficiency_angle=8
    )

    d_kite3 = Deflector("kite3", cl=1, cl_range=(0.4, 0.9), area=12, efficiency_angle=4)
    # d_fish3 = Deflector(
    #     "fish3", cl=0.7, cl_range=(0.2, 0.4), area=0.07, efficiency_angle=8
    # )

    fk1 = FishKite("fk1", wind_speed_i, rising_angle_1, fish=d_fish1, kite=d_kite1)
    fk2 = FishKite("fk2", wind_speed_i, rising_angle_2, fish=d_fish2, kite=d_kite2)
    fk3 = FishKite("fk3", wind_speed_i, rising_angle_2, fish=d_fish2, kite=d_kite3)

    proj1 = Project([fk1, fk2])
    proj2 = Project([fk1, fk2, fk3])

    # %%
    fig2 = plot_cases(
        list_of_cases=[fk1, fk2], draw_ortho_grid=False, add_background_image=False
    )
    fig2.show()
    # %%

    fig1 = proj1.plot(add_background_image=True)
    fig1.show()

    # %%
    fig2 = proj2.plot(draw_ortho_grid=False, add_background_image=True)
    fig2.show()
    # %%
    print(f"{d_kite2.glide_ratio() =:.3f}")
    print(f"{d_fish2.glide_ratio() =:.3f}")
    print(f"{d_kite2.projected_efficiency_angle(43) =:.3f}")
    print(f"{d_fish2.projected_efficiency_angle(43) =:.3f}")
    print(f"- fisk kite-")
    print(f"{fk2.projected_efficiency_angle('kite') =:.3f}")
    print(f"{fk2.total_efficiency() =:.3f}")
    print("----")
    print(f"{fk2.fluid_velocity_ratio() =}")
    vr = fk2.fluid_velocity_ratio()
    print(f"{fk2.true_wind_angle(vr) =}")
    print(f"{fk2.apparent_wind(vr) =}")
    print(f"{fk2.apparent_watter(vr) =}")

# %%


# 32 , 265
# 700 , 636

# %%
