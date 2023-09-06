# %% [markdown]
# # FishPy
# ##  Test the model.py results
#
# V1 - fred - 05/08/2023
#
# Program write to be test with pytest library
#
# Install this library and run the program in the terminal using : pytest test_model.py
# %% Import
import datetime

import os
import model as m
from model import Deflector, FishKite, Project
import pandas as pd
import pickle
from pytest import fixture

cwd = os.getcwd()
# %% test fk


@fixture(scope="session")
def fk1():
    wind_speed_i = 15  # kt
    rising_angle_1 = 33  # deg

    d_kite1 = Deflector(
        "kite1", cl=0.4, cl_range=(0.4, 0.9), flat_area=24, efficiency_angle=12
    )
    d_fish1 = Deflector(
        "fish1", cl=0.2, cl_range=(0.2, 0.4), flat_area=0.1, efficiency_angle=14
    )
    fk1 = FishKite("fk1", wind_speed_i, rising_angle_1, fish=d_fish1, kite=d_kite1)
    return fk1


@fixture(scope="session")
def fk2():
    wind_speed_i = 15  # kt
    rising_angle_2 = 20  # deg

    d_kite2 = Deflector(
        "kite2", cl=0.6, cl_range=(0.3, 0.9), flat_area=25, efficiency_angle=4
    )
    d_fish2 = Deflector(
        "fish2", cl=0.4, cl_range=(0.2, 0.4), flat_area=0.07, efficiency_angle=8
    )
    fk2 = FishKite("fk2", wind_speed_i, rising_angle_2, fish=d_fish2, kite=d_kite2)
    return fk2


@fixture(scope="session")
def fk3():
    wind_speed_i = 23  # kt
    rising_angle_2 = 50  # deg

    d_kite3 = Deflector(
        "kite3", cl=1, cl_range=(0.4, 0.9), flat_area=12, efficiency_angle=4
    )
    d_fish3 = Deflector(
        "fish2", cl=0.4, cl_range=(0.2, 0.4), flat_area=0.1, efficiency_angle=45
    )
    fk3 = FishKite("fk2", wind_speed_i, rising_angle_2, fish=d_fish3, kite=d_kite3)
    return fk3


# %%
def test_cable_tension(fk1, fk2, fk3):
    assert fk1.cable_tension() == 137.28886487924606
    assert fk2.cable_tension() == 793.8592843467501
    assert fk3.cable_tension() == 130.75025597711434


def test_apparent_water(fk1, fk2, fk3):
    assert fk1.apparent_water() == 22.496637897925087
    assert fk2.apparent_water() == 45.72015638713615
    assert fk3.apparent_water() == 15.524093042426685


def test_flight_creation_and_add(fk1, fk2, fk3):
    proj = Project([fk1, fk2, fk3])
    df = proj.perf_table()

    path_ref = os.path.join(cwd, "test_assets", "projet123.pkl")
    # df.to_pickle(path_ref)
    df_ref = pd.read_pickle(path_ref)
    pd.testing.assert_frame_equal(df, df_ref)


# %%
