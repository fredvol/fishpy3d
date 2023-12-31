#####################################
## Model to simulate Fish kite VPP ##
## 27/07/2023 - frd                ##
#####################################

# Notes:
#     All angles are store in degrees and convert in rad inside the functions.
# %%  impot librairies
# from bdb import effective
import os
import numpy as np
import pandas as pd

import plotly.express as px

# import plotly.figure_factory as ff
# import plotly.offline as po
import plotly.graph_objects as go

import jsonpickle

from fish_plot_3d import plot_3d_cases, plot_3d_cases_risingangle, plot_side_view

# import scipy.optimize as opt

# %% constants
GRAVITY = 9.81  # m/s-2
RHO_AIR = 1.29  # kg/m3
RHO_WATER = 1025  # kg/m3
CONV_KTS_MS = 0.5144456333854638  # (m/s)/kt
# CABLE_STRENGTH_MM2 = 100  # daN/mm2

data_folder = os.path.join(os.path.dirname(__file__), "data")

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


def load_fish_kite(m_path):
    return FishKite.from_json(m_path)


def cosspace_extra_angle(start: float = 0.0, stop: float = 90, cos_coef: float = 0.5):
    """
    Makes a cosine-spaced vector.
    Assuming average step = 1 !
    Args:
        start: Value to start at.
        end: Value to end at.
        cos_coef: coeficient to apply the cosinus.
    """
    num = stop - start + 1
    step_length = [1 - np.cos(np.radians(i * 180 / num)) * cos_coef for i in range(num)]
    result = [start]
    for i in range(num - 1):
        result.append(result[-1] + step_length[i + 1])

    # Fix the endpoints, which might not be exactly right due to floating-point error.
    result[0] = start
    result[-1] = stop

    return result


def perf_table(df_big, target_wind):
    df_perf = (
        df_big[
            (df_big["isValid"])
            & (df_big["true_wind_calculated_kt_rounded"] == target_wind)
        ]
        .groupby("fk_name")
        .agg(
            Max_apparent_water_kt=("apparent_water_kt", "max"),
            VMG_Upwind=("vmg_y_kt", "max"),
            VMG_Downwind=("vmg_y_kt", "min"),
        )
        .T
    )
    return df_perf


def perf_table_general(df_big, proj):
    list_dfi = []
    for fk in proj.lst_fishkite:
        dfi = df_big[df_big["fk_name"] == fk.name]
        dfi_valid = dfi[dfi["isValid"]]

        # wind range
        min_TW = round(dfi_valid["true_wind_calculated_kt"].min(), 1)
        max_TW = round(dfi_valid["true_wind_calculated_kt"].max(), 1)

        # pct valid
        true_count = dfi["isValid"].sum()
        total_count = len(dfi)

        if total_count == 0:
            percentage = "NA"
        else:
            percentage = round((true_count / total_count) * 100, 1)

        dict_stats = {
            "cable_diam_mm": round(fk.cable_diameter() * 1000, 2),
            "true_wind_range": f"[{min_TW}:{max_TW}]",
            "pct_of_OPvalid": f"{percentage} %",
        }

        dfi = pd.DataFrame(dict_stats, index=[fk.name])

        list_dfi.append(dfi)

    df_perf_G = pd.concat(list_dfi)
    return df_perf_G


# %% Class
class Pilot:
    """class holding all infomation about the pilot"""

    # init method or constructor
    def __init__(
        self,
        mass: float,  # kg
        pilot_drag: float,  # S*Cxp (m²)
    ):
        self.mass = mass
        self.pilot_drag = pilot_drag
        self.obj_version = 1

    def weight(self):  # Newton
        return self.mass * GRAVITY

    def to_dict(self):
        """Converts the Pilot object to a dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def to_json(self, filename):
        data_json = jsonpickle.encode(self)
        with open(filename, "w") as json_file:
            json_file.write(data_json)

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str)


# %%


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
        self.obj_version = 1

    # load save

    def to_json(self, filename):
        data_json = jsonpickle.encode(self)
        with open(filename, "w") as json_file:
            json_file.write(data_json)

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str)

    # computation

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
        cable_length_fish_unstreamline: float,
        cable_length_fish_streamline: float,
        cable_length_kite: float,
        cable_strength: float,
        cx_cable_water_unstreamline: float,
        cx_cable_water_streamline: float,
        cx_cable_air: float,
        tip_fish_depth: float,
        cable_strength_mm2: float,
    ):
        self.name = name
        self.wind_speed = wind_speed * CONV_KTS_MS  # input in kt converted in m/s
        self.rising_angle = rising_angle  # deg
        self.fish = fish
        self.kite = kite
        self.pilot = pilot
        self._extra_angle = extra_angle  # deg  this will later been computed
        self.cable_length_fish_unstreamline = cable_length_fish_unstreamline  # m
        self.cable_length_fish_streamline = cable_length_fish_streamline  # m
        self.cable_length_kite = cable_length_kite  # m
        self.cable_strength = cable_strength  # DaN
        self.cx_cable_water_unstreamline = cx_cable_water_unstreamline  # no unit
        self.cx_cable_water_streamline = cx_cable_water_streamline  # no unit
        self.cx_cable_air = cx_cable_air  # no unit
        self.tip_fish_depth = tip_fish_depth  # no
        self.cable_strength_mm2 = cable_strength_mm2  # DaN/mm2

        self.obj_version = 1

    def __repr__(self):
        return f"FishKite({self.name}): wind_speed[kt]:{ms_to_knot(self.wind_speed)} rising_angle:{self.rising_angle}  \n Kite:{self.kite}  \n Fish:{self.fish}"

    # Load and save
    def to_json_str(self):
        return jsonpickle.encode(self)

    def to_json(self, filename):
        """Convert to object to json

        Note: Once the Json is produce, the '__main__.FishKite' should be remplace by 'model3d.FishKite' by hand. to be loaded by orther files
        Args:
            filename (str): filename to write it
        """

        data_json = jsonpickle.encode(self)
        with open(filename, "w") as json_file:
            json_file.write(data_json)
        print("ok exported")

    @classmethod
    def from_json(cls, filename, classes=None):
        with open(filename, "r") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str, classes=classes)

    @classmethod
    def from_json_str(cls, json_str, classes=None):
        return jsonpickle.decode(json_str, classes=classes)

    # geometry

    def cable_diameter(self):
        return (
            np.sqrt((self.cable_strength * 4) / (np.pi * self.cable_strength_mm2))
            / 1000
        )

    def fish_center_depth(self):  # m
        return self.tip_fish_depth + self.fish.projected_span() * 0.5 * np.cos(
            np.radians(self.rising_angle)
        )

    def cable_length_in_water(self):  # m
        return self.fish_center_depth() / np.sin(np.radians(self.rising_angle))

    def cable_length_fish(self):  # m
        return self.cable_length_fish_streamline + self.cable_length_fish_unstreamline

    def cable_length_in_air(self):  # m
        return self.cable_length_fish() - self.cable_length_in_water()

    def cable_length_in_water_streamline(self):  # m
        if self.cable_length_fish_streamline < self.cable_length_in_water():
            return self.cable_length_fish_streamline
        else:
            return self.cable_length_in_water()

    def cable_length_in_water_unstreamline(self):  # m
        if self.cable_length_fish_streamline < self.cable_length_in_water():
            return self.cable_length_in_water() - self.cable_length_fish_streamline
        else:
            return 0

    # drag
    def cable_water_drag(self):  # m2
        return (
            self.cable_length_in_water_streamline()
            * self.cable_diameter()
            * self.cx_cable_water_streamline
            + self.cable_length_in_water_unstreamline()
            * self.cable_diameter()
            * self.cx_cable_water_unstreamline
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
        x_pilot = 0 + self.cable_length_fish() * np.cos(np.radians(self.rising_angle))
        y_pilot = -self.fish_center_depth + self.cable_length_fish() * np.sin(
            np.radians(self.rising_angle)
        )
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

    def apparent_water_ms(self):  # m/s
        apparent_water_ms = np.sqrt(
            self.fish_total_force()
            / (self.fish_c_force() * 0.5 * RHO_WATER * self.fish.projected_area())
        )
        return apparent_water_ms

    def apparent_water_wind_angle(self):
        # Apparent water / wind (degrees. = Boat course relative to wind)
        angle = np.arccos(
            (
                self.apparent_water_ms() ** 2
                - self.apparent_wind_ms() ** 2
                + self.true_wind_calculated() ** 2
            )
            / (2 * self.apparent_water_ms() * self.true_wind_calculated())
        )
        return 180 - np.degrees(angle)

    def apparent_wind_wind_angle(self):  # deg
        return -self.total_efficiency() + self.apparent_water_wind_angle()

    def vmg_x(self):  # m/s
        return self.apparent_water_ms() * np.sin(
            np.radians(self.apparent_water_wind_angle())
        )

    def vmg_y(self):  # m/s
        return self.apparent_water_ms() * np.cos(
            np.radians(self.apparent_water_wind_angle())
        )

    def vmg_x_kt(self):  # kt
        return ms_to_knot(self.vmg_x())

    def vmg_y_kt(self):  # kt ; + = upwind)
        return ms_to_knot(self.vmg_y())

    def true_wind_calculated(self):  # m/s
        return np.sqrt(
            self.apparent_water_ms() ** 2
            + self.apparent_wind_ms() ** 2
            - 2
            * self.apparent_water_ms()
            * self.apparent_wind_ms()
            * np.cos(np.radians(self.total_efficiency()))
        )

    # def find_raising_angle(self):
    #     fun = lambda x: ((self.modified_extra_angle(x) - self.wind_speed) ** 2) ** 0.5
    #     # results = opt.minimize_scalar(fun, bounds=(0.1, 89), method='bounded')
    #     results = opt.minimize(fun, x0=(30), bounds=((0.1), (89)))
    #     return results

    def create_df(self, nb_points=20):
        """Create df with all the data

        Args:
            nb_points (int, optional): nb points to divide the CL range ( fish and kite). Defaults to 20.

        Returns:
            DataFrame: all the data.
        """
        range_fish = np.linspace(
            self.fish.cl_range["min"], self.fish.cl_range["max"], nb_points
        ).round(3)
        range_kite = np.linspace(
            self.kite.cl_range["min"], self.kite.cl_range["max"], nb_points
        ).round(3)
        # range_fish = [round(i, 3) for i in range_fish]  #slow
        # range_kite = [round(i, 3) for i in range_kite]  #slow
        range_rising_angle = [
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
        ]  # was generated before by :#  [1] + list(np.arange(5, 90, 5))
        # range_extra_angle = np.arange(2, 90, 1)
        range_extra_angle = cosspace_extra_angle(2, 84, 0.5)

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

        # reduce momery size
        df["rising_angle"] = df["rising_angle"].astype("int32")

        # add the simplify criteria
        # balance_range = []
        df["simplify"] = False
        for i in range(nb_points):
            # balance_range.append((range_kite[i], range_fish[-(i + 1)]))
            df.loc[
                (df["kite_cl"] == range_kite[i])
                & (df["fish_cl"] == range_fish[-(i + 1)]),
                "simplify",
            ] = True

        df[f"kite_roll_angle"] = df[f"rising_angle"] + df["extra_angle"]
        # df.drop(df[df["kite_roll_angle"] >= 90].index, inplace=True)
        df = df[df["kite_roll_angle"] < 90]

        # df["rising_angle_rad"] = df["rising_angle"].apply(np.radians)
        # df["extra_angle_rad"] = df["extra_angle"].apply(np.radians)
        # df["kite_roll_angle_rad"] = df["kite_roll_angle"].apply(np.radians)

        df["rising_angle_rad"] = np.radians(df["rising_angle"])
        df["extra_angle_rad"] = np.radians(df["extra_angle"])
        df["kite_roll_angle_rad"] = np.radians(df["kite_roll_angle"])

        # cable
        df[
            "fish_center_depth"
        ] = self.tip_fish_depth + self.fish.projected_span() * 0.5 * np.cos(
            df["rising_angle_rad"]
        )
        df["cable_length_in_water"] = df["fish_center_depth"] / np.sin(
            df["rising_angle_rad"]
        )

        df["cable_length_in_water_streamline"] = np.where(
            (self.cable_length_fish_streamline < df["cable_length_in_water"]),
            self.cable_length_fish_streamline,
            df["cable_length_in_water"],
        )

        df["cable_length_in_water_unstreamline"] = np.where(
            (self.cable_length_fish_streamline < df["cable_length_in_water"]),
            (df["cable_length_in_water"] - self.cable_length_fish_streamline),
            0,
        )

        df["cable_water_drag"] = (
            df["cable_length_in_water_streamline"]
            * self.cable_diameter()
            * self.cx_cable_water_streamline
        ) + (
            df["cable_length_in_water_unstreamline"]
            * self.cable_diameter()
            * self.cx_cable_water_unstreamline
        )

        df["cable_length_in_air"] = (
            self.cable_length_fish() - df["cable_length_in_water"]
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
        df["proj_efficiency_water_LD"] = (
            df[f"fish_lift"] / df[f"total_water_drag"]
        ) * np.cos(df["rising_angle_rad"])
        df["projected_efficiency_water_rad"] = np.arctan(
            1 / (df["proj_efficiency_water_LD"])
        )

        df["proj_efficiency_air_LD"] = (
            df[f"kite_lift"] / df[f"total_air_drag"]
        ) * np.cos(df["kite_roll_angle_rad"])
        df["kite_projected_efficiency_rad"] = np.arctan(
            1 / (df["proj_efficiency_air_LD"])
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

        # df["flat_power_ratio"] = (
        #     RHO_AIR * df["kite_c_force"] * self.kite.projected_area()
        # ) / (RHO_WATER * df["fish_c_force"] * self.fish.projected_area())

        # df["projected_power_ratio"] = (
        #     df["flat_power_ratio"]
        #     * (np.cos(df[f"kite_roll_angle_rad"]))
        #     / (np.cos(df[f"rising_angle_rad"]))
        # )

        # position ( pilot , kite)

        df["y_pilot"] = 0 + self.cable_length_fish() * np.cos(df["rising_angle_rad"])
        df["z_pilot"] = -df["fish_center_depth"] + self.cable_length_fish() * np.sin(
            df["rising_angle_rad"]
        )

        df["y_kite"] = df["y_pilot"] + self.cable_length_kite * np.cos(
            df["kite_roll_angle_rad"]
        )
        df["z_kite"] = df["z_pilot"] + self.cable_length_kite * np.sin(
            df["kite_roll_angle_rad"]
        )

        # speeds
        df["apparent_water_ms"] = np.sqrt(
            df["fish_total_force"]
            / (df["fish_c_force"] * 0.5 * RHO_WATER * self.fish.projected_area())
        )

        df["apparent_water_kt"] = df["apparent_water_ms"] / CONV_KTS_MS

        df["apparent_wind_ms"] = np.sqrt(
            df["kite_total_force"]
            / (df["kite_c_force"] * 0.5 * RHO_AIR * self.kite.projected_area())
        )

        df["apparent_wind_kt"] = df["apparent_wind_ms"] / CONV_KTS_MS

        df["true_wind_calculated"] = np.sqrt(
            df["apparent_water_ms"] ** 2
            + df["apparent_wind_ms"] ** 2
            - 2
            * df["apparent_water_ms"]
            * df["apparent_wind_ms"]
            * np.cos(df["total_efficiency_rad"])
        )

        df["true_wind_calculated_kt"] = df["true_wind_calculated"] / CONV_KTS_MS

        df["apparent_water_wind_rad"] = np.pi - np.arccos(
            (
                df["apparent_water_ms"] ** 2
                - df["apparent_wind_ms"] ** 2
                + df["true_wind_calculated"] ** 2
            )
            / (2 * df["apparent_water_ms"] * df["true_wind_calculated"])
        )

        df["vmg_x"] = df["apparent_water_ms"] * np.sin(df["apparent_water_wind_rad"])
        df["vmg_y"] = df["apparent_water_ms"] * np.cos(df["apparent_water_wind_rad"])

        df["vmg_x_kt"] = df["vmg_x"] / CONV_KTS_MS
        df["vmg_y_kt"] = df["vmg_y"] / CONV_KTS_MS

        # cable strength
        df["cable_strength_margin"] = self.cable_strength / (
            df["fish_total_force"] / 10
        )
        df["cable_break"] = df["cable_strength_margin"] <= 1

        # group data
        df["true_wind_calculated_kt_rounded"] = (
            df["true_wind_calculated_kt"] / 2
        ).round() * 2  # round to even numbers

        cavitation_conditions = [
            (df["apparent_water_kt"] > 40)
            | ((df["fish_total_force"] / self.fish.flat_area()) > 50000),
        ]
        df["cavitation"] = np.select(cavitation_conditions, [True], default=False)

        df["isValid"] = ~(df["cavitation"] | df["cable_break"])

        conditions = [
            (df["cable_break"] & df["cavitation"]),
            df["cable_break"],
            df["cavitation"],
            True,  # Default condition
        ]

        choices = ["broken cavitated", "break", "cavitation", "no failure"]

        df["failure"] = np.select(conditions, choices, default=0)

        # %optimise size - was saving 10mo ( 113mb instead 122mb) but loose precision.
        # df["kite_cl"] = df["kite_cl"].astype(np.float32)
        # df["fish_cl"] = df["fish_cl"].astype(np.float32)
        # df["rising_angle"] = df["rising_angle"].astype(np.int16)
        # df["extra_angle"] = df["extra_angle"].astype(np.int16)
        # df["kite_roll_angle"] = df["kite_roll_angle"].astype(np.int32)

        return df


class Project:
    """Project holding different FishKites, help to plot several FK"""

    def __init__(self, lst_fishkite=[], name="Project1"):
        self.name = name
        self.lst_fishkite = lst_fishkite
        self.obj_version = 1

    def __str__(self):
        return f"{self.name}"

        # Load and save

    def to_json_str(self):
        return jsonpickle.encode(self)

    def detail(self):
        detail_str = f"Project contains {len(self.lst_fishkite)} FiskKite(s):"
        for i in self.lst_fishkite:
            detail_str += "\n-\n"
            detail_str += str(i)
        return detail_str

    def create_df(self):
        df_list = []

        for fk in self.lst_fishkite:
            dfi = fk.create_df()
            dfi["fk_name"] = fk.name
            df_list.append(dfi)

        df = pd.concat(df_list, ignore_index=True)
        # add general index  to find back the data
        df["indexG"] = df.index
        return df

        return fig


# %%


# %%# Parameter
if __name__ == "__main__":
    fk1_file = os.path.join(data_folder, "saved_fk1.json")
    fk2_file = os.path.join(data_folder, "saved_fk2.json")

    fk1 = FishKite.from_json(fk1_file)
    fk2 = FishKite.from_json(fk2_file)

    proj = Project([fk1, fk2])

    df = fk1.create_df()

    # %%
    dfM = proj.create_df()

    # %%
    df1 = dfM[dfM["fk_name"] == "fk1"]

    # %% performance compare
    dfx = (
        dfM[dfM["isValid"]]
        .groupby(["fk_name", "true_wind_calculated_kt_rounded"])
        .agg({"vmg_y_kt": ["min", "max"]})
        .reset_index()
    )
    dfx.columns = [" ".join(col).strip() for col in dfx.columns.values]

    # %% Display maneuvrability graph
    colors = ["rgba(26,150,65,0.5)", "rgba(100,149,237,0.6)"]

    fig = go.Figure()

    fig.update_layout(
        title="Maneuvrability",
        xaxis_title="True wind",
        yaxis_title="VMG_Y",
        legend_title="Fishkite",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    for (
        i,
        f,
    ) in enumerate(dfx["fk_name"].unique()):
        print(f"{f=}")

        dfxi = dfx[dfx["fk_name"] == f]
        fig.add_trace(
            go.Scatter(
                x=dfxi["true_wind_calculated_kt_rounded"],
                y=dfxi["vmg_y_kt min"],
                fill=None,
                mode="lines",
                line_color=colors[i],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dfxi["true_wind_calculated_kt_rounded"],
                y=dfxi["vmg_y_kt max"],
                fill="tonexty",  # fill area between trace0 and trace1
                fillcolor=colors[i],
                legendgroup=f,
                mode="lines",
                line_color=colors[i],
            )
        )

    fig.show()

    # %%
    what = "fk_name"
    height_size = 700
    px.scatter(
        dfM[dfM["fk_name"] == "fk2"],
        x="vmg_x_kt",
        y="vmg_y_kt",
        color=what,
        hover_data=[
            "apparent_water_ms",
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
        "find_raising_angle",
        "data_to_plot_polar",
        "perf_table",
        "true_wind_angle",
        "create_df",
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
    dfcheck = df[
        (df["fish_cl"] == 0.474)
        & (df["kite_cl"] == 0.811)
        & (df["rising_angle"] == 40)
        & (df["extra_angle"] == 20)
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
