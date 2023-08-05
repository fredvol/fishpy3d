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

cwd = os.getcwd()
# test fk


def test_flight_creation_and_add():
    """
    Create from 0
    """
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

    proj = Project([fk1, fk2, fk3])
    df = proj.perf_table()

    path_ref = os.path.join(cwd, "test_assets", "projet123.pkl")
    df_ref = pd.read_pickle(path_ref)
    pd.testing.assert_frame_equal(df, df_ref)
