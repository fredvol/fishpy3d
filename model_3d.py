#####################################
## Model to simulate Fish kite VPP ##
## 27/07/2023 - frd                ##
#####################################

# Notes:
#     All angles are store in degrees and convert in rad inside the functions.
# %%  impot librairies
from bdb import effective
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as po
import plotly.graph_objects as go
from itertools import product

from fish_plot_3d import plot_3d_cases, plot_3d_cases_risingangle

import scipy.optimize as opt

# %% constants
GRAVITY = 9.81  # m/s-2
RHO_AIR = 1.29  # kg/m3
RHO_WATER = 1025  # kg/m3
CONV_KTS_MS = 0.5144456333854638  # (m/s)/kt
CABLE_STRENGTH_MM2 = 100  # daN/mm2


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


def ms_to_knot(value):
    return value / CONV_KTS_MS


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
        profil_drag_coeff: float,
        parasite_drag_pct: float,
    ):
        self.name = name
        self.cl = cl  # no unit
        self._flat_area = flat_area  # m2
        self.flat_ratio = flat_ratio  # no unit
        self.flat_aspect_ratio = flat_aspect_ratio  # no unit
        self.profil_drag_coeff = profil_drag_coeff  # no unit
        self._parasite_drag_pct = parasite_drag_pct  # % of flat area

        self.cl_range = {"min": cl_range[0], "max": cl_range[1]}  # no unit

    def flat_area(self):  # m2
        return self._flat_area

    def projected_area(self):  # m2
        return self._flat_area * self.flat_ratio

    def flat_span(self):  # m
        return np.sqrt(self.flat_aspect_ratio * self.projected_area())

    def projected_span(self):  # m
        return self.flat_span() * self.flat_ratio

    def lift(self):  # S*cl
        return self.projected_area() * self.cl

    def profile_drag(self):
        return self.profil_drag_coeff * self.flat_area()

    def parasite_drag(self):
        return self._parasite_drag_pct * self.flat_area() + self.profile_drag()

    def induced_drag(self):  # S*Cx
        # In table
        return (
            self.cl**2 / (np.pi * self.flat_aspect_ratio * 0.95)
        ) * self.flat_area()

    def total_drag(self):  # S*Cx
        return self.parasite_drag() + self.induced_drag()

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


# class Kite(Deflector):
#     """This classe inherit from deflector and simulate a kite with a pilot."""

#     def __init__(
#         self,
#         name: str,
#         cl: float,
#         cl_range: tuple,
#         flat_area: float,
#         flat_ratio: float,
#         flat_aspect_ratio: float,
#         profil_drag_coeff: float,
#         parasite_drag_pct: float,
#         pilot=Pilot,
#     ):
#         super(self.__class__, self).__init__(
#             name,
#             cl,
#             cl_range,
#             flat_area,
#             flat_ratio,
#             flat_aspect_ratio,
#             profil_drag_coeff,
#             parasite_drag_pct,
#         )

#         self.pilot = pilot

#     def total_drag_with_pilot(self):  # S*Cx
#         return self.total_drag() + self.pilot.pilot_drag

#     def efficiency_LD(self):
#         return self.lift() / self.total_drag_with_pilot()

#     def c_force(self):
#         return (
#             np.sqrt(self.total_drag_with_pilot() ** 2 + self.lift() ** 2)
#             / self.projected_area()
#         )

#     def efficiency_angle(self):  # deg1
#         return np.degrees(np.arctan(1 / self.efficiency_LD()))


class FishKite:
    """This is the central object, most of the computation are done here"""

    def __init__(
        self,
        name: str,
        wind_speed: float,
        rising_angle: float,
        fish: Deflector,
        kite: Deflector,
        pilot: Pilot,
        extra_angle: float,
        cable_length_fish: float,
        cable_length_kite: float,
        cable_strength: float,
        cx_cable_water: float,
        cx_cable_air: float,
        tip_fish_depth: float,
    ):
        self.name = name
        self.wind_speed = wind_speed * CONV_KTS_MS  # input in kt converted in m/s
        self.rising_angle = rising_angle  # deg
        self.fish = fish
        self.kite = kite
        self.pilot = pilot
        self._extra_angle = extra_angle  # deg  this will later been computed
        self.cable_length_fish = cable_length_fish  # m
        self.cable_length_kite = cable_length_kite  # m
        self.cable_strength = cable_strength  # DaN
        self.cx_cable_water = cx_cable_water  # no unit
        self.cx_cable_air = cx_cable_air  # no unit
        self.tip_fish_depth = tip_fish_depth  # no

    def __repr__(self):
        return f"FishKite({self.name}): wind_speed[kt]:{ms_to_knot(self.wind_speed)} rising_angle:{self.rising_angle}  \n Kite:{self.kite}  \n Fish:{self.fish}"

    # geometry

    def cable_diameter(self):
        return np.sqrt((self.cable_strength * 4) / (np.pi * CABLE_STRENGTH_MM2)) / 1000

    def fish_center_depth(self):  # m
        return self.tip_fish_depth + self.fish.projected_span() * 0.5 * np.cos(
            np.radians(self.rising_angle)
        )

    def cable_length_in_water(self):  # m
        return self.fish_center_depth() / np.sin(np.radians(self.rising_angle))

    def cable_length_in_air(self):  # m
        return self.cable_length_fish - self.cable_length_in_water()

    # drag
    def cable_water_drag(self):  # m2
        return (
            self.cable_length_in_water() * self.cable_diameter() * self.cx_cable_water
        )

    def cable_air_drag(self):  # m2
        return self.cable_length_in_air() * self.cable_diameter() * self.cx_cable_air

    def total_water_drag(self):  # m2
        return self.cable_water_drag() + self.fish.total_drag()

    def total_air_drag(self):  # m2
        return self.cable_air_drag() + self.kite.total_drag() + self.pilot.pilot_drag

    # efficiency
    def efficiency_water_LD(self):
        return self.fish.lift() / self.total_water_drag()

    def efficiency_air_LD(self):
        return self.kite.lift() / self.total_air_drag()

    def efficiency_water_deg(self):
        return np.degrees(np.arctan(1 / self.efficiency_water_LD()))

    def projected_efficiency_water_deg(self):
        # in table
        return np.degrees(
            np.arctan(
                1 / (self.efficiency_water_LD() * np.cos(np.radians(self.rising_angle)))
            )
        )

    def efficiency_air_deg(self):
        return np.degrees(np.arctan(1 / self.efficiency_air_LD()))

    # coeff force
    def kite_c_force(self):  # No unit
        return (
            np.sqrt(self.total_air_drag() ** 2 + self.kite.lift() ** 2)
            / self.kite.projected_area()
        )

    def fish_c_force(self):  # No unit
        return (
            np.sqrt(self.total_water_drag() ** 2 + self.fish.lift() ** 2)
            / self.fish.projected_area()
        )

    def flat_power_ratio(self):  # air/water
        flat_pwr_ratio = (
            RHO_AIR * self.kite_c_force() * self.kite.projected_area()
        ) / (RHO_WATER * self.fish_c_force() * self.fish.projected_area())
        return flat_pwr_ratio

    #  now considering extra angle
    def extra_angle(self):  # deg
        return self._extra_angle

    def position_pilot(self):
        x_pilot = self.cable_length_fish * np.cos(np.radians(self.rising_angle))
        y_pilot = self.cable_length_fish * np.sin(np.radians(self.rising_angle))
        return (x_pilot, y_pilot)

    def position_kite(self):
        x_pilot, y_pilot = self.position_pilot()
        x_kite = x_pilot + self.cable_length_kite * np.cos(
            np.radians(self.kite_roll_angle())
        )
        y_kite = y_pilot + self.cable_length_kite * np.sin(
            np.radians(self.kite_roll_angle())
        )
        return (x_kite, y_kite)

    def kite_roll_angle(self):
        return self.rising_angle + self.extra_angle()

    def kite_projected_efficiency_angle(self):  # deg
        return np.degrees(
            np.arctan(
                1
                / (
                    np.cos(np.radians(self.kite_roll_angle()))
                    * self.efficiency_air_LD()
                )
            )
        )

    def total_efficiency(self):  # deg
        return (
            self.kite_projected_efficiency_angle()
            + self.projected_efficiency_water_deg()
        )

    def fish_total_force(self):  # N
        """geometrical resolution of the 3 forces"""
        return (
            self.pilot.weight()
            * np.sin(np.radians(90 - self.kite_roll_angle()))
            / np.sin(np.radians(self.extra_angle()))
        )

    def kite_total_force(self):  # N
        """geometrical resolution of the 3 forces"""
        return (
            self.pilot.weight()
            * np.sin(np.radians(90 - self.rising_angle))
            / np.sin(np.radians(self.extra_angle()))
        )

    def projected_power_ratio(self):  # air/water
        return (
            self.flat_power_ratio()
            * (np.cos(np.radians(self.kite_roll_angle())))
            / (np.cos(np.radians(self.rising_angle)))
        )

    def apparent_wind_ms(self):
        apparent_wind_ms = np.sqrt(
            self.kite_total_force()
            / (self.kite_c_force() * 0.5 * RHO_AIR * self.kite.projected_area())
        )
        return apparent_wind_ms

    def apparent_watter_ms(self):  # m/s
        apparent_water_ms = np.sqrt(
            self.fish_total_force()
            / (self.fish_c_force() * 0.5 * RHO_WATER * self.fish.projected_area())
        )
        return apparent_water_ms

    def apparent_water_wind_angle(self):
        # Apparent water / wind (degrees. = Boat course relative to wind)
        angle = np.arccos(
            (
                self.apparent_watter_ms() ** 2
                - self.apparent_wind_ms() ** 2
                + self.true_wind_calculated() ** 2
            )
            / (2 * self.apparent_watter_ms() * self.true_wind_calculated())
        )
        return 180 - np.degrees(angle)

    def apparent_wind_wind_angle(self):  # deg
        return -self.total_efficiency() + self.apparent_water_wind_angle()

    def vmg_x(self):  # m/s
        return self.apparent_watter_ms() * np.sin(
            np.radians(self.apparent_water_wind_angle())
        )

    def vmg_y(self):  # m/s
        return self.apparent_watter_ms() * np.cos(
            np.radians(self.apparent_water_wind_angle())
        )

    def vmg_x_kt(self):  # kt
        return ms_to_knot(self.vmg_x())

    def vmg_y_kt(self):  # kt ; + = upwind)
        return ms_to_knot(self.vmg_y())

    def true_wind_calculated(self):  # m/s
        return np.sqrt(
            self.apparent_watter_ms() ** 2
            + self.apparent_wind_ms() ** 2
            - 2
            * self.apparent_watter_ms()
            * self.apparent_wind_ms()
            * np.cos(np.radians(self.total_efficiency()))
        )

    def speed_gap_modified_extra_angle(self, angle, debug=True):
        self._extra_angle = angle
        if debug:
            print(angle)

        return self.true_wind_calculated() - self.wind_speed

    def find_raising_angle(self):
        fun = lambda x: ((self.modified_extra_angle(x) - self.wind_speed) ** 2) ** 0.5
        # results = opt.minimize_scalar(fun, bounds=(0.1, 89), method='bounded')
        results = opt.minimize(fun, x0=(30), bounds=((0.1), (89)))
        return results

    ################# BELOW THAT ALL IS FOR THE 2D ##################

    # def fluid_velocity_ratio(self):
    #     current_ratio = (
    #         RHO_AIR
    #         * self.kite.flat_area()
    #         * self.kite.cl
    #         / (RHO_WATER * self.fish.flat_area() * self.fish.cl)
    #     ) ** 0.5
    #     return current_ratio

    # def fluid_velocity_ratio_range(self):
    #     min_ratio = (
    #         RHO_AIR
    #         * self.kite.flat_area()
    #         * self.kite.cl_range["min"]
    #         / (RHO_WATER * self.fish.flat_area() * self.fish.cl_range["max"])
    #     ) ** 0.5
    #     max_ratio = (
    #         RHO_AIR
    #         * self.kite.flat_area()
    #         * self.kite.cl_range["max"]
    #         / (RHO_WATER * self.fish.flat_area() * self.fish.cl_range["min"])
    #     ) ** 0.5

    #     return {"max": max_ratio, "min": min_ratio}

    # def true_wind_angle(self, velocity_ratio):
    #     """From 2D calculation might be obsolete"""
    #     # TODO to clean
    #     value = np.degrees(
    #         np.arctan(
    #             np.sin(np.radians(self.total_efficiency()))
    #             / (velocity_ratio - np.cos(np.radians(self.total_efficiency())))
    #         )
    #     )
    #     if value > 0:
    #         return 180 - value
    #     else:
    #         return 180 - (180 + value)

    def create_df(self, nb_points=10):
        """Create df with all the data

        Args:
            nb_points (int, optional): nb points to divide the CL range ( fish and kite). Defaults to 20.

        Returns:
            DataFrame: all the data.
        """
        range_fish = np.linspace(
            self.fish.cl_range["min"], self.fish.cl_range["max"], nb_points
        )
        range_kite = np.linspace(
            self.kite.cl_range["min"], self.kite.cl_range["max"], nb_points
        )
        range_fish = [round(i, 3) for i in range_fish]
        range_kite = [round(i, 3) for i in range_kite]
        range_rising_angle = range_rising_angle = [1] + list(np.arange(5, 90, 5))
        range_extra_angle = np.arange(2, 90, 1)

        ##  previous data generation method  ( slow 15s)
        # dfa = pd.DataFrame(
        #     list(product(range_kite, range_fish, range_rising_angle, range_extra_angle)),
        #     columns=["kite_cl", "fish_cl", "rising_angle", "extra_angle"],
        # )

        # Generate all combinations using NumPy broadcasting
        kite_cl, fish_cl, rising_angle, extra_angle = np.meshgrid(
            range_kite, range_fish, range_rising_angle, range_extra_angle
        )
        df = pd.DataFrame(
            {
                "kite_cl": kite_cl.ravel(),  # The .ravel() function is used to flatten the arrays into 1D arrays
                "fish_cl": fish_cl.ravel(),
                "rising_angle": rising_angle.ravel(),
                "extra_angle": extra_angle.ravel(),
            }
        )

        # add the simplify criteria
        balance_range = []
        for i in range(nb_points):
            balance_range.append((range_kite[i], range_fish[-(i + 1)]))
            df.loc[
                (df["kite_cl"] == range_kite[i])
                & (df["fish_cl"] == range_fish[-(i + 1)]),
                "simplify",
            ] = 1

        df[f"kite_roll_angle"] = df[f"rising_angle"] + df["extra_angle"]
        df.drop(df[df["kite_roll_angle"] >= 90].index, inplace=True)

        df["rising_angle_rad"] = df["rising_angle"].apply(np.radians)
        df["extra_angle_rad"] = df["extra_angle"].apply(np.radians)
        df["kite_roll_angle_rad"] = df["kite_roll_angle"].apply(np.radians)

        # cable
        df[
            "fish_center_depth"
        ] = self.tip_fish_depth + self.fish.projected_span() * 0.5 * np.cos(
            df["rising_angle_rad"]
        )
        df["cable_length_in_water"] = df["fish_center_depth"] / np.sin(
            df["rising_angle_rad"]
        )
        df["cable_length_in_air"] = self.cable_length_fish - df["cable_length_in_water"]
        df["cable_water_drag"] = (
            df["cable_length_in_water"] * self.cable_diameter() * self.cx_cable_water
        )
        df["cable_air_drag"] = (
            df["cable_length_in_air"] * self.cable_diameter() * self.cx_cable_air
        )

        # lift and drag
        df["fish_lift"] = self.fish.projected_area() * df["fish_cl"]
        df["kite_lift"] = self.kite.projected_area() * df["kite_cl"]

        df["fish_induced_drag"] = (
            df["fish_cl"] ** 2 / (np.pi * self.fish.flat_aspect_ratio * 0.95)
        ) * self.fish.flat_area()
        df["kite_induced_drag"] = (
            df["kite_cl"] ** 2 / (np.pi * self.kite.flat_aspect_ratio * 0.95)
        ) * self.kite.flat_area()

        df["total_water_drag"] = (
            df[f"fish_induced_drag"]
            + self.fish.parasite_drag()
            + df["cable_water_drag"]
        )
        df["total_air_drag"] = (
            df[f"kite_induced_drag"]
            + self.kite.parasite_drag()
            + df["cable_air_drag"]
            + self.pilot.pilot_drag
        )

        df["fish_c_force"] = (
            np.sqrt(df["total_water_drag"] ** 2 + df["fish_lift"] ** 2)
            / self.fish.projected_area()
        )
        df["kite_c_force"] = (
            np.sqrt(df["total_air_drag"] ** 2 + df["kite_lift"] ** 2)
            / self.kite.projected_area()
        )

        # For extra angle
        df["projected_efficiency_water_rad"] = np.arctan(
            1
            / (
                (df[f"fish_lift"] / df[f"total_water_drag"])
                * np.cos(df["rising_angle_rad"])
            )
        )

        df["kite_projected_efficiency_rad"] = np.arctan(
            1
            / (
                np.cos(df["kite_roll_angle_rad"])
                * (df[f"kite_lift"] / df[f"total_air_drag"])
            )
        )

        df["total_efficiency_rad"] = (
            df[f"projected_efficiency_water_rad"] + df[f"kite_projected_efficiency_rad"]
        )

        df["fish_total_force"] = (
            self.pilot.weight()
            * np.sin((np.pi / 2) - df[f"kite_roll_angle_rad"])
            / np.sin(df[f"extra_angle_rad"])
        )  # N
        df["kite_total_force"] = (
            self.pilot.weight()
            * np.sin((np.pi / 2) - df[f"rising_angle_rad"])
            / np.sin(df[f"extra_angle_rad"])
        )  # N

        df["flat_power_ratio"] = (
            RHO_AIR * df["kite_c_force"] * self.kite.projected_area()
        ) / (RHO_WATER * df["fish_c_force"] * self.fish.projected_area())

        df["projected_power_ratio"] = (
            df["flat_power_ratio"]
            * (np.cos(df[f"kite_roll_angle_rad"]))
            / (np.cos(df[f"rising_angle_rad"]))
        )

        # speeds
        df["apparent_watter_ms"] = np.sqrt(
            df["fish_total_force"]
            / (df["fish_c_force"] * 0.5 * RHO_WATER * self.fish.projected_area())
        )

        df["apparent_wind_ms"] = np.sqrt(
            df["kite_total_force"]
            / (df["kite_c_force"] * 0.5 * RHO_AIR * self.kite.projected_area())
        )

        df["true_wind_calculated"] = np.sqrt(
            df["apparent_watter_ms"] ** 2
            + df["apparent_wind_ms"] ** 2
            - 2
            * df["apparent_watter_ms"]
            * df["apparent_wind_ms"]
            * np.cos(df["total_efficiency_rad"])
        )

        df["true_wind_calculated_kt"] = df["true_wind_calculated"] / CONV_KTS_MS

        df["apparent_water_wind_rad"] = np.pi - np.arccos(
            (
                df["apparent_watter_ms"] ** 2
                - df["apparent_wind_ms"] ** 2
                + df["true_wind_calculated"] ** 2
            )
            / (2 * df["apparent_watter_ms"] * df["true_wind_calculated"])
        )

        df["vmg_x"] = df["apparent_watter_ms"] * np.sin(df["apparent_water_wind_rad"])
        df["vmg_y"] = df["apparent_watter_ms"] * np.cos(df["apparent_water_wind_rad"])

        df["vmg_x_kt"] = df["vmg_x"] / CONV_KTS_MS
        df["vmg_y_kt"] = df["vmg_y"] / CONV_KTS_MS

        # cable strength
        df["cable_strength_margin"] = self.cable_strength / (
            df["fish_total_force"] / 10
        )
        df["cable_break"] = df["cable_strength_margin"] <= 1

        # group data
        df["true_wind_calculated_kt_rounded"] = df["true_wind_calculated_kt"].round()

        return df

    # def data_to_plot_polar(self):
    #     # OLD FOR 2D VERSION
    #     """Prepare the detail to be plot

    #     Returns: Dict
    #     """
    #     vr = self.fluid_velocity_ratio()
    #     current_apparent_watter_pct = self.apparent_watter(vr) / self.wind_speed * 100
    #     current_true_wind_angle = self.true_wind_angle(vr)

    #     anchor = [0, 0]
    #     wind = [0, -100]
    #     op_point = pol2cart(current_apparent_watter_pct, current_true_wind_angle)
    #     polar_pts = self.compute_polar()
    #     watter_speed_kt = self.apparent_watter()

    #     return {
    #         "anchor": anchor,
    #         "wind": wind,
    #         "op_point": op_point,
    #         "polar_pts": polar_pts,
    #         "watter_speed_kt": watter_speed_kt,
    #     }

    # def plot(self, draw_ortho_grid=True, add_background_image=False):
    #     # OLD FOR 2D VERSION
    #     """Apply the plot_case function to this FishKite"""
    #     fig = plot_cases([self], draw_ortho_grid, add_background_image)
    #     return fig

    # def add_plot_elements(self, fig, m_color=None, add_legend_name=False):
    #     ##OLD FOR 2D VERSION
    #     """Add elements to a already existing plotly figure

    #     Args:
    #         fig (Plotly figure): The figure to modify
    #         m_color (str, optional): Hex string containing the color to use. Defaults to None.
    #         add_legend_name (bool, optional): If True the Fishkite name is added to the figure legende. Defaults to False.

    #     Returns:
    #         _type_: _description_
    #     """
    #     data_plot = self.data_to_plot_polar()
    #     df_polar = data_plot["polar_pts"]

    #     def generate_hover_text(row):
    #         return (
    #             f"{row['name']}: {round(row['apparent_watter_kt'],1)} kts {row['note']}"
    #         )

    #     def generate_marker_size(row):
    #         return 9 if len(row["note"]) else 1

    #     df_polar["text_hover"] = df_polar.apply(generate_hover_text, axis=1)
    #     df_polar["marker_size"] = df_polar.apply(generate_marker_size, axis=1)
    #     legend_name = ""
    #     if add_legend_name:
    #         legend_name = "_" + self.name

    #     # add speed wind label /!\ warning if different fishkite wind speed
    #     fig.add_annotation(
    #         x=-10,
    #         y=-50,
    #         text=f"True Wind:{round(ms_to_knot(self.wind_speed,1))} kt",
    #         showarrow=False,
    #         font=dict(color="red", size=12),
    #         textangle=-90,
    #     )

    #     # add polar
    #     fig.add_trace(
    #         (
    #             go.Scatter(
    #                 x=df_polar["x_watter_pct"],
    #                 y=df_polar["y_watter_pct"],
    #                 # mode="lines",
    #                 legendgrouptitle_text=self.name,
    #                 name=f"Polar{legend_name}",
    #                 text=df_polar["text_hover"],
    #                 marker_size=df_polar["marker_size"],
    #                 mode="lines+markers",
    #                 hoverinfo="text",
    #                 line=dict(
    #                     color=m_color,
    #                     width=3,
    #                 ),
    #             )
    #         )
    #     )

    #     # trajectory  ( TODO chage to  fig.add_shape(type="line",) for label  )
    #     fig.add_trace(
    #         add_line(
    #             data_plot["anchor"],
    #             data_plot["op_point"],
    #             m_name=f"Water speed {legend_name} ",
    #             group_name=self.name,
    #             extra_dict=dict(dash="dash", color=m_color),
    #         )
    #     )

    #     fig.add_annotation(
    #         x=data_plot["op_point"][0],
    #         y=data_plot["op_point"][1],
    #         text=f'{round(data_plot["watter_speed_kt"],1)} kts',
    #         showarrow=True,
    #         xanchor="center",
    #         arrowhead=1,
    #         font=dict(color=m_color, size=12),
    #         arrowcolor=m_color,
    #         arrowsize=0.3,
    #         bgcolor="#ffffff",
    #         bordercolor=m_color,
    #         borderwidth=2,
    #         ax=10,
    #         ay=-30,
    #     )

    #     # Apparent_wind_vector
    #     fig.add_trace(
    #         add_line(
    #             data_plot["wind"],
    #             data_plot["op_point"],
    #             m_name=f"Apparent wind speed {legend_name}",
    #             group_name=self.name,
    #             extra_dict=dict(dash="dot", color=m_color),
    #         )
    #     )

    #     return fig


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

    def create_df(self):
        df_list = []

        for fk in self.lst_fishkite:
            dfi = fk.create_df()
            dfi["fk_name"] = fk.name
            df_list.append(dfi)

        df = pd.concat(df_list, ignore_index=True)
        return df

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
    d_pilot = Pilot(mass=80, pilot_drag=0.25)
    d_kite1 = Deflector(
        "kite1",
        cl=0.8,
        cl_range=(0.4, 1),
        flat_area=20,
        flat_ratio=0.85,
        flat_aspect_ratio=6,
        profil_drag_coeff=0.013,
        parasite_drag_pct=0.03,  # 0.69,
    )
    d_fish1 = Deflector(
        "fish1",
        cl=0.6,
        cl_range=(0.2, 0.6),
        flat_area=0.1,
        flat_ratio=0.64,
        profil_drag_coeff=0.01,
        flat_aspect_ratio=8.5,
        parasite_drag_pct=0.06,
    )

    fk1 = FishKite(
        "fk1",
        wind_speed=15,
        rising_angle=20,
        fish=d_fish1,
        kite=d_kite1,
        pilot=d_pilot,
        extra_angle=20,
        cable_length_fish=30,
        cable_length_kite=12,
        cable_strength=500,
        cx_cable_water=1,
        cx_cable_air=1,
        tip_fish_depth=0.5,
    )

    d_kite2 = Deflector(
        "kite2",
        cl=0.8,
        cl_range=(0.1, 0.6),
        flat_area=35,
        flat_ratio=0.85,
        flat_aspect_ratio=10,
        profil_drag_coeff=0.013,
        parasite_drag_pct=0.01,  # 0.69,
    )
    d_fish2 = Deflector(
        "fish2",
        cl=0.6,
        cl_range=(0.2, 1),
        flat_area=0.3,
        flat_ratio=0.64,
        profil_drag_coeff=0.01,
        flat_aspect_ratio=10,
        parasite_drag_pct=0.02,
    )

    fk2 = FishKite(
        "fk2",
        wind_speed=15,
        rising_angle=20,
        fish=d_fish2,
        kite=d_kite2,
        pilot=d_pilot,
        extra_angle=20,
        cable_length_fish=30,
        cable_length_kite=12,
        cable_strength=500,
        cx_cable_water=1,
        cx_cable_air=1,
        tip_fish_depth=0.5,
    )

    proj = Project([fk1, fk2])

    dfM = proj.create_df()

    # %%
    what = "fk_name"
    height_size = 700
    px.scatter(
        dfM[dfM["fk_name"] == "fk2"],
        x="vmg_x_kt",
        y="vmg_y_kt",
        color=what,
        hover_data=[
            "apparent_watter_ms",
            what,
            "fish_total_force",
            "cable_strength_margin",
        ],
        height=height_size,
        width=height_size * 1.1,
    )

    # %% display all for fish
    print(" ALL DATA FOR FISH")
    attributes = dir(d_fish1)

    # Iterate through the attributes and call only the callable ones (functions)
    for attr_name in attributes:
        if "__" in attr_name:
            continue

        attr = getattr(d_fish1, attr_name)
        if callable(attr):
            result = attr()
            print(f" {attr_name},\t : {result}")
        else:
            print(f" {attr_name},\t : {attr}")
    # %% display alll for kite
    print(" ALL DATA FOR KITE")
    attributes = dir(d_kite1)

    # Iterate through the attributes and call only the callable ones (functions)
    for attr_name in attributes:
        if "__" in attr_name:
            continue

        attr = getattr(d_kite1, attr_name)
        if callable(attr):
            result = attr()
            print(f" {attr_name},\t : {result}")
        else:
            print(f" {attr_name},\t : {attr}")

    # %% display alll for fishkite
    print(" ALL DATA FOR FISH Kite")
    attributes = dir(fk1)
    list_to_exclude = [
        "add_plot_elements",
        "plot",
        "compute_polar",
        "speed_gap_modified_extra_angle",
        "find_raising_angle",
        "data_to_plot_polar",
        "perf_table",
        "true_wind_angle",
    ]
    # Iterate through the attributes and call only the callable ones (functions)
    for attr_name in attributes:
        if "__" in attr_name or attr_name in list_to_exclude:
            continue

        attr = getattr(fk1, attr_name)
        if callable(attr):
            result = attr()
            print(f" {attr_name},\t : {result}")
        else:
            print(f" {attr_name},\t : {attr}")

    # %%

    # %% Create double range
    df = fk1.create_df()

    # %%
    # assert df
    # df.to_pickle("dfall.pkl")

    df_ref = pd.read_pickle("dfall.pkl")
    # pd.testing.assert_frame_equal(df, df_ref)

    # %%
    dfcheck = df[
        (df["fish_cl"] == 0.474)
        & (df["kite_cl"] == 0.811)
        & (df["rising_angle"] == 40)
        & (df["extra_angle"] == 10)
    ]

    dfcheck.T
    # %% PLOT 3D

    # fig = px.scatter_3d(df, x='vmg_x', y='vmg_y', z='rising_angle',
    #               color='true_wind_calculated')
    # fig.show()
    # %%  One rising angle

    fig = px.scatter(
        df[(df["rising_angle"] == 20)],
        x="vmg_x",
        y="vmg_y",
        color="true_wind_calculated",
    )
    fig.show()

    # %%
    target_wind = 23
    target_rising_angle = 20

    dfr = df[(df["rising_angle"] == target_rising_angle)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dfr["vmg_x_kt"],
            y=dfr["vmg_y_kt"],
            mode="markers",
            marker=dict(
                size=2,
                # I want the color to be green if
                # lower_limit ≤ y ≤ upper_limit
                # else red
                color=((dfr["true_wind_calculated_kt_rounded"] == target_wind)).astype(
                    "int"
                ),
                colorscale=[[0, "blue"], [1, "red"]],
            ),
        )
    )

    fig.update_layout(
        title=f"Polar pts for rising angle:{target_rising_angle} : red TW= {target_wind} kt",
        xaxis_title="vmg_x_kt",
        yaxis_title="vmg_y_kt",
        legend_title="Legend Title",
    )
    fig.show()

    # %% Plot selected

    dfs = df[
        (df["rising_angle"] == target_rising_angle)
        & (df["true_wind_calculated_kt_rounded"] == target_wind)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dfs["vmg_x_kt"],
            y=dfs["vmg_y_kt"],
            mode="markers",
            marker=dict(
                size=4,
                # I want the color to be green if
                # lower_limit ≤ y ≤ upper_limit
                # else red
                color=((dfs["simplify"] == 1)).astype("int"),
                colorscale=[[0, "red"], [1, "green"]],
            ),
        )
    )

    fig.update_layout(
        title=f"Polar pts for rising angle:{target_rising_angle} : red TW= {target_wind} kt",
        xaxis_title="vmg_x_kt",
        yaxis_title="vmg_y_kt",
        legend_title="Legend Title",
    )
    fig.show()

    # %%  only selected points
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
    fig.show()
    # %%
    # df_kt = df[df["true_wind_calculated_kt_rounded"] == target_wind]
    # fig = px.density_contour(
    #     dfs,
    #     x="vmg_x_kt",
    #     y="vmg_y_kt",
    #     z="rising_angle",
    #     color="rising_angle",
    #     title=f"Polar contour for TW={target_wind} kt",
    # )

    # dfsimplify = dfs[dfs["simplify"] == 1]
    # fig.add_trace(
    #     go.Scatter(
    #         x=dfsimplify["vmg_x_kt"],
    #         y=dfsimplify["vmg_y_kt"],
    #         mode="markers",
    #         marker=dict(
    #             size=8,
    #             # I want the color to be green if
    #             # lower_limit ≤ y ≤ upper_limit
    #             # else red
    #             color="red",
    #         ),
    #     )
    # )
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    # )

    # fig.show()
    # %% Validation pivot table

    # validation_rising_angle = 30

    # validation_extra_angle = 20
    # dfv = df[
    #     (df["fish_cl"].isin(range_fish))
    #     & (df["kite_cl"].isin(range_kite))
    #     & (df["rising_angle"] == validation_rising_angle)
    #     & (df["extra_angle"] == validation_extra_angle)
    # ]

    # pd.pivot_table(
    #     dfv,
    #     values="true_wind_calculated_kt",
    #     index=["kite_cl"],
    #     columns=["fish_cl"],
    #     aggfunc=np.sum,
    # )
    # %%

    dfx = df[df["true_wind_calculated_kt_rounded"] == 30]
    fig = px.scatter(
        dfx,
        x="vmg_x_kt",
        y="vmg_y_kt",
        color="rising_angle",  # "extra_angle",
        title=f"Polar pts",
    )
    fig.show()
    # %%
    from scipy.spatial import ConvexHull, convex_hull_plot_2d

    rng = np.random.default_rng()

    points = dfx[["vmg_x_kt", "vmg_y_kt"]].to_numpy()  # 30 random points in 2-D

    hull = ConvexHull(points)
    # %%

    import matplotlib.pyplot as plt

    plt.plot(points[:, 0], points[:, 1], "o")

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "k-")
    # %%
