#####################################
## Model to simulate Fish kite VPP ##
## 27/07/2023 - frd                ##
#####################################
# %%  impot librairies

import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from fish_plot import plot_cases, add_line

# %% constants
GRAVITY = 9.81  # m/s-2
RHO_AIR = 1.29  # kg/m3
RHO_WATER = 1025  # kg/m3
CONV_KTS_MS = 0.514444


# %%  Functions


def cart2pol(x, y):
    """Carthesian coordinates to Polar coordinates"""

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi, start=90):
    """Polar coordinates to Carthesian coordinates"""
    phirad = np.radians(start - phi)
    x = rho * np.cos(phirad)
    y = rho * np.sin(phirad)
    return (x, y)


# %% Class
class Pilot:
    """class holding all infomation about the pilot"""

    # init method or constructor
    def __init__(
        self,
        mass: float,  # kg
        pilot_drag: tuple,  # S*Cxp (m²)
    ):
        self.mass = mass
        self.pilot_drag = pilot_drag

    def weight(self):  # Newton
        return self.mass * GRAVITY


class Deflector:
    """All object wich deviate a flux"""

    def __init__(
        self,
        name: str,
        cl: float,
        cl_range: tuple,
        flat_area: float,
        flat_ratio: float,
        flat_aspect_ratio: float,
        parasite_drag_pct: float,
        efficiency_angle: float,
        line_length: float,
    ):
        self.name = name
        self.cl = cl  # no unit
        self._flat_area = flat_area  # m2
        self.flat_ratio = flat_ratio  # m2
        self.flat_aspect_ratio = flat_aspect_ratio  # m2
        self._parasite_drag_pct = parasite_drag_pct  # % of flat area
        self._efficiency_angle = efficiency_angle  # deg
        self.line_length = line_length  # m

        self.cl_range = {"min": cl_range[0], "max": cl_range[1]}  # no unit

    def flat_area(self):  # m2
        return self._flat_area

    def projected_area(self):  # m2
        return self._flat_area * self.flat_ratio

    def flat_span(self):  # m
        return np.sqrt(self.flat_aspect_ratio * self.projected_area())

    def lift(self):  # S*cl
        return self.projected_area() * self.cl

    def parasite_drag(self):
        return self._parasite_drag_pct * self.flat_area()

    def induced_drag(self):  # S*Cx
        return (
            self.cl**2 / (np.pi * self.flat_aspect_ratio * 0.95)
        ) * self.flat_area()

    def total_drag(self):  # S*Cx
        return self._parasite_drag + self.induced_drag()

    def c_force(self):  # No unit
        return (
            np.sqrt(self.total_drag() ** 2 + self.lift() ** 2) / self.projected_area()
        )

    def efficiency_LD(self):  # no unit
        return self.lift() / self.total_drag()

    def efficiency_angle(self):  # deg
        return np.degrees(np.arctan(1 / self.efficiency_LD()))

    def __repr__(self):
        return f"{self.name} Cl:{self.cl} efficiency_angle:{self.efficiency_angle()} Area:{self.flat_area()} "

    # Sample Method
    # def glide_ratio(self) -> float:
    #     return 1 / np.tan(np.radians(self.efficiency_angle()))


class Kite(Deflector):
    """This classe inherit from deflector and simulate a kite with a pilot."""

    def __init__(
        self,
        name: str,
        cl: float,
        cl_range: tuple,
        flat_area: float,
        flat_ratio: float,
        flat_aspect_ratio: float,
        parasite_drag_pct: float,
        efficiency_angle: float,
        line_length: float,
        pilot=Pilot,
    ):
        super(self.__class__, self).__init__(
            name,
            cl,
            cl_range,
            flat_area,
            flat_ratio,
            flat_aspect_ratio,
            parasite_drag_pct,
            efficiency_angle,
            line_length,
        )

        self.pilot = pilot

    def total_drag_with_pilot(self):  # S*Cx
        return self.total_drag() + self.pilot.pilot_drag

    def efficiency_LD(self):
        return self.lift() / self.total_drag_with_pilot()

    def c_force(self):
        return (
            np.sqrt(self.total_drag_with_pilot() ** 2 + self.lift() ** 2)
            / self.projected_area()
        )

    def efficiency_angle(self):  # deg1
        return np.degrees(np.arctan(1 / self.efficiency_LD()))


class FishKite:
    """This is the central object, most of the computation are done here"""

    def __init__(
        self,
        name: str,
        wind_speed: float,
        rising_angle: float,
        fish: Deflector,
        kite: Deflector,
        extra_angle: float,
    ):
        self.name = name
        self.wind_speed = wind_speed  # kt
        self.rising_angle = rising_angle  # deg
        self.fish = fish
        self.kite = kite
        self._extra_angle = extra_angle  # deg  this will later been computed

    def __repr__(self):
        return f"FishKite({self.name}): wind_speed:{self.wind_speed} rising_angle:{self.rising_angle}  \n Kite:{self.kite}  \n Fish:{self.fish}"

    def extra_angle(self):  # deg
        return self._extra_angle

    def kite_roll_angle(self):
        return self.rising_angle + self.extra_angle()

    def projected_efficiency_angle(self, what: str) -> float:
        # including extra angle for Kite
        if what == "kite":
            return np.degrees(
                np.arctan(
                    1
                    / (
                        np.cos(np.radians(self.kite_roll_angle()))
                        * self.kite.efficiency_LD()
                    )
                )
            )
        elif what == "fish":
            return np.degrees(
                np.arctan(
                    1
                    / (
                        np.cos(np.radians(self.rising_angle))
                        * self.fish.efficiency_LD()
                    )
                )
            )
        else:
            print(f" {what} is unkown , waiting for 'fish' or 'kite' ")

    def total_efficiency(self):
        return self.projected_efficiency_angle(
            "kite"
        ) + self.projected_efficiency_angle("fish")

    def flat_power_ratio(self):  # air/water
        flat_pwr_ratio = (
            RHO_AIR * self.kite.c_force() * self.kite.projected_area()
        ) / (RHO_WATER * self.fish.c_force() * self.fish.projected_area())
        return flat_pwr_ratio

    def projected_power_ratio(self):  # air/water
        return (
            self.flat_power_ratio()
            * (np.cos(np.radians(self.kite_roll_angle())))
            / (np.cos(np.radians(self.rising_angle)))
        )

    def fluid_velocity_ratio(self):
        current_ratio = (
            RHO_AIR
            * self.kite.flat_area()
            * self.kite.cl
            / (RHO_WATER * self.fish.flat_area() * self.fish.cl)
        ) ** 0.5
        return current_ratio

    def fluid_velocity_ratio_range(self):
        min_ratio = (
            RHO_AIR
            * self.kite.flat_area()
            * self.kite.cl_range["min"]
            / (RHO_WATER * self.fish.flat_area() * self.fish.cl_range["max"])
        ) ** 0.5
        max_ratio = (
            RHO_AIR
            * self.kite.flat_area()
            * self.kite.cl_range["max"]
            / (RHO_WATER * self.fish.flat_area() * self.fish.cl_range["min"])
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
        """This fuunction was used on the 2D version and will be obsolete,"""
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()
        apparent_wind_kt = (
            self.wind_speed
            * np.sin(np.radians(180 - self.true_wind_angle(velocity_ratio)))
            / np.sin(np.radians(self.total_efficiency()))
        )
        return apparent_wind_kt

    def apparent_watter(self, velocity_ratio=None):
        """This fuunction was used on the 2D version and will be obsolete,"""
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()

        apparent_water_kt = self.fluid_velocity_ratio() * self.apparent_wind(
            velocity_ratio
        )
        return apparent_water_kt

    def fish_total_force(self):
        """geometrical resolution of the 3 forces"""
        return self.kite.pilot.weight()

    def cable_tension(self, velocity_ratio=None):
        """This fuunction was used on the 2D version and will be obsolete,
        it assume assume the watter speed is already been compute
        """
        if velocity_ratio is None:
            velocity_ratio = self.fluid_velocity_ratio()
        cable_tension_dan = (
            0.5
            * RHO_WATER
            * self.fish.flat_area()
            * self.fish.cl
            * (self.apparent_watter(velocity_ratio) * CONV_KTS_MS) ** 2
            / 10  # to convert to DaN
        )
        return cable_tension_dan

    def compute_polar(self, nb_value=70):
        """Compute the polar for several (cl_fish , cl_kite) ratio

        Args:
            nb_value (int, optional): Nb of point to compute for the polar. Defaults to 70.

        Returns:
            DataFrame: Dataframe result.
        """
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
        """Collect different data on the fishkite Operating point and general point on the polar

        Returns:
            DataFrame
        """
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
        """Prepare the detail to be plot

        Returns: Dict
        """
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
        """Apply the plot_case function to this FishKite"""
        fig = plot_cases([self], draw_ortho_grid, add_background_image)
        return fig

    def add_plot_elements(self, fig, m_color=None, add_legend_name=False):
        """Add elements to a already existing plotly figure

        Args:
            fig (Plotly figure): The figure to modify
            m_color (str, optional): Hex string containing the color to use. Defaults to None.
            add_legend_name (bool, optional): If True the Fishkite name is added to the figure legende. Defaults to False.

        Returns:
            _type_: _description_
        """
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
    """Project holding different FishKites, help to plot several FK"""

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
    rising_angle_1 = 20  # deg

    d_pilot = Pilot("pilot1", mass=80, pilot_drag=0.49)
    d_kite1 = Kite(
        "kite1",
        cl=0.8,
        cl_range=(0.4, 0.9),
        flat_area=18,
        flat_ratio=0.85,
        flat_aspect_ratio=6,
        parasite_drag_pct=0.038555556,  # 0.69,
        efficiency_angle=12,
        line_length=12,
        pilot=d_pilot,
    )
    d_fish1 = Deflector(
        "fish1",
        cl=0.6,
        cl_range=(0.2, 0.4),
        flat_area=0.1,
        flat_ratio=0.64,
        flat_aspect_ratio=8.5,
        parasite_drag_pct=0.0650,
        efficiency_angle=14,
        line_length=30,
    )

    fk1 = FishKite(
        "fk1", wind_speed_i, rising_angle_1, fish=d_fish1, kite=d_kite1, extra_angle=20
    )
    # %%
    print(f"{d_kite1.efficiency_LD() =:.3f}")
    print(f"{d_fish1.efficiency_LD() =:.3f}")

    print(f"- fisk kite-")
    print(f"{fk1.projected_efficiency_angle('kite') =:.3f}")
    print(f"{fk1.projected_efficiency_angle('fish') =:.3f}")
    print(f"{fk1.total_efficiency() =:.3f}")
    print("----")
    print(f"{fk1.fluid_velocity_ratio() =}")
    vr = fk1.fluid_velocity_ratio()
    print(f"{fk1.true_wind_angle(vr) =}")
    print(f"{fk1.apparent_wind(vr) =}")
    print(f"{fk1.apparent_watter(vr) =}")

    # %%
    # proj1 = Project([fk1, fk2])
    # proj2 = Project([fk1, fk2, fk3])

    # %%
    df_polar = fk1.data_to_plot_polar()["polar_pts"]
    # %%
    fig2 = plot_cases(
        list_of_cases=[fk1],
        draw_ortho_grid=True,
        draw_iso_speed=False,
        add_background_image=True,
        height_size=800,
    )
    fig2.show()
    # %%


# %%


# 32 , 265
# 700 , 636

# %%
