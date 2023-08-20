# %% FISH Kite model
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from fish_plot import plot_cases, add_line

# %% constants
RHO_AIR = 1.29  # kg/m3
RHO_WATER = 1025  # kg/m3
CONV_KTS_MS = 0.514444


# %%  Functions


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi, start=90):
    phirad = np.radians(start - phi)
    x = rho * np.cos(phirad)
    y = rho * np.sin(phirad)
    return (x, y)


# %% Class


class Deflector:
    # init method or constructor
    def __init__(
        self,
        name: str,
        cl: float,
        cl_range: tuple,
        flat_area: float,
        efficiency_angle: float,
    ):
        self.name = name
        self.cl = cl
        self.area = flat_area  # m2
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

    def apparent_wind(self, velocity_ratio=None):
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()
        apparent_wind_kt = (
            self.wind_speed
            * np.sin(np.radians(180 - self.true_wind_angle(velocity_ratio)))
            / np.sin(np.radians(self.total_efficiency()))
        )
        return apparent_wind_kt

    def apparent_watter(self, velocity_ratio=None):
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()

        apparent_water_kt = self.fluid_velocity_ratio() * self.apparent_wind(
            velocity_ratio
        )
        return apparent_water_kt

    def cable_tension(self, velocity_ratio=None):
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()
        cable_tension_dan = (
            0.5
            * RHO_WATER
            * self.fish.area
            * self.fish.cl
            * (self.apparent_watter(velocity_ratio) * CONV_KTS_MS) ** 2
            / 10  # to convert to DaN
        )
        return cable_tension_dan

    def compute_polar(self, nb_value=70):
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
        df_polar["y_watter_kt"] = df_polar["apparent_watter_kt"] * np.cos(
            np.radians(df_polar["true_wind_angle"])
        )

        df_polar["name"] = self.name

        # add special points
        df_polar["note"] = ""
        df_polar.loc[
            df_polar["apparent_watter_kt"].idxmax(), "note"
        ] += " Max Watter_speed"
        df_polar.loc[df_polar["y_watter_kt"].idxmax(), "note"] += " Vmg UpW"
        df_polar.loc[df_polar["y_watter_kt"].idxmin(), "note"] += " Vmg downW"
        return df_polar

    def perf_table(self):
        df = self.compute_polar()
        dict_stats = {
            "Total Efficiency [°]": self.total_efficiency(),
            "Max Water speed [kt]": df["apparent_watter_kt"].max(),
            "VMG UpWind [kt]": df["y_watter_kt"].max(),
            "VMG DownWind [kt]": df["y_watter_kt"].min(),
            "OP Water speed [kt]": self.apparent_watter(),
            "OP Cable tension [DaN]": self.cable_tension(),
        }
        return pd.DataFrame(dict_stats, index=[self.name])

    def data_to_plot_polar(self):
        vr = self.fluid_velocity_ratio()
        current_apparent_watter_pct = self.apparent_watter(vr) / self.wind_speed * 100
        current_true_wind_angle = self.true_wind_angle(vr)

        anchor = [0, 0]
        wind = [0, -100]
        op_point = pol2cart(current_apparent_watter_pct, current_true_wind_angle)
        polar_pts = self.compute_polar()
        watter_speed_kt = self.apparent_watter()

        return {
            "anchor": anchor,
            "wind": wind,
            "op_point": op_point,
            "polar_pts": polar_pts,
            "watter_speed_kt": watter_speed_kt,
        }

    def plot(self, draw_ortho_grid=True, add_background_image=False):
        fig = plot_cases([self], draw_ortho_grid, add_background_image)
        return fig

    def add_plot_elements(self, fig, m_color=None, add_legend_name=False):
        data_plot = self.data_to_plot_polar()
        df_polar = data_plot["polar_pts"]

        def generate_hover_text(row):
            return (
                f"{row['name']}: {round(row['apparent_watter_kt'],1)} kts {row['note']}"
            )

        def generate_marker_size(row):
            return 9 if len(row["note"]) else 1

        df_polar["text_hover"] = df_polar.apply(generate_hover_text, axis=1)
        df_polar["marker_size"] = df_polar.apply(generate_marker_size, axis=1)
        legend_name = ""
        if add_legend_name:
            legend_name = "_" + self.name

        # add speed wind label /!\ warning if different fishkite wind speed
        fig.add_annotation(
            x=-10,
            y=-50,
            text=f"True Wind:{round(self.wind_speed,1)} kt",
            showarrow=False,
            font=dict(color="red", size=12),
            textangle=-90,
        )

        # add polar
        fig.add_trace(
            (
                go.Scatter(
                    x=df_polar["x_watter_pct"],
                    y=df_polar["y_watter_pct"],
                    # mode="lines",
                    legendgrouptitle_text=self.name,
                    name=f"Polar{legend_name}",
                    text=df_polar["text_hover"],
                    marker_size=df_polar["marker_size"],
                    mode="lines+markers",
                    hoverinfo="text",
                    line=dict(
                        color=m_color,
                        width=3,
                    ),
                )
            )
        )

        # trajectory  ( TODO chage to  fig.add_shape(type="line",) for label  )
        fig.add_trace(
            add_line(
                data_plot["anchor"],
                data_plot["op_point"],
                m_name=f"Water speed {legend_name} ",
                group_name=self.name,
                extra_dict=dict(dash="dash", color=m_color),
            )
        )

        fig.add_annotation(
            x=data_plot["op_point"][0],
            y=data_plot["op_point"][1],
            text=f'{round(data_plot["watter_speed_kt"],1)} kts',
            showarrow=True,
            xanchor="center",
            arrowhead=1,
            font=dict(color=m_color, size=12),
            arrowcolor=m_color,
            arrowsize=0.3,
            bgcolor="#ffffff",
            bordercolor=m_color,
            borderwidth=2,
            ax=10,
            ay=-30,
        )

        # Apparent_wind_vector
        fig.add_trace(
            add_line(
                data_plot["wind"],
                data_plot["op_point"],
                m_name=f"Apparent wind speed {legend_name}",
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
        detail_str = f"Project contains {len(self.lst_fishkite)} FiskKite(s):"
        for i in self.lst_fishkite:
            detail_str += "\n-\n"
            detail_str += str(i)
        return detail_str

    def perf_table(self):
        list_df = [fk.perf_table() for fk in self.lst_fishkite]
        df = pd.concat(list_df)
        return df.T.reset_index()

    def plot(self, draw_ortho_grid=True, add_background_image=False):
        fig = plot_cases(
            self.lst_fishkite,
            draw_ortho_grid,
            draw_iso_speed=True,
            add_background_image=add_background_image,
        )

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
    df_polar = fk1.data_to_plot_polar()["polar_pts"]
    # %%
    fig2 = plot_cases(
        list_of_cases=[fk1, fk2],
        draw_ortho_grid=True,
        draw_iso_speed=False,
        add_background_image=True,
        height_size=800,
    )
    fig2.show()
    # %%

    fig1 = proj1.plot(add_background_image=False)
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
