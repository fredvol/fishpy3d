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
from model_3d import Deflector, FishKite, Pilot
import pandas as pd
import pickle
from pytest import fixture
import test_assets.data_test_model3d as data_test_model3d

cwd = os.getcwd()
test_folder = os.path.join(cwd, "test_assets")


# %% function
def object_2_dict(m_object):
    attributes = dir(m_object)
    dict_data = {}

    # Iterate through the attributes and call only the callable ones (functions)
    for attr_name in attributes:
        if "__" in attr_name:
            continue

        attr = getattr(m_object, attr_name)
        if callable(attr):
            result = attr()
            dict_data[attr_name] = result

        else:
            dict_data[attr_name] = attr

    return dict_data


# %%


@fixture(scope="session")
def fk1():
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
    return fk1


@fixture(scope="session")
def fk2():
    d_pilot2 = Pilot(mass=90, pilot_drag=0.35)
    d_kite2 = Deflector(
        "kite2",
        cl=0.8,
        cl_range=(0.3, 0.8),
        flat_area=18,
        flat_ratio=0.85,
        flat_aspect_ratio=8,
        profil_drag_coeff=0.017,
        parasite_drag_pct=0.04,  # 0.69,
    )
    d_fish2 = Deflector(
        "fish2",
        cl=0.6,
        cl_range=(0.25, 0.7),
        flat_area=0.15,
        flat_ratio=0.64,
        profil_drag_coeff=0.01,
        flat_aspect_ratio=9.5,
        parasite_drag_pct=0.08,
    )

    fk2 = FishKite(
        "fk2",
        wind_speed=25,
        rising_angle=30,
        fish=d_fish2,
        kite=d_kite2,
        pilot=d_pilot2,
        extra_angle=10,
        cable_length_fish=35,
        cable_length_kite=15,
        cable_strength=700,
        cx_cable_water=1,
        cx_cable_air=1,
        tip_fish_depth=0.9,
    )
    return fk2


# %%
def test_fk1_object(fk1):
    fk1_fish = object_2_dict(fk1.fish)
    fk1_kite = object_2_dict(fk1.kite)

    assert fk1_fish == data_test_model3d.data_fk1["fk1_fish"]
    assert fk1_kite == data_test_model3d.data_fk1["fk1_kite"]


# %%
def test_fk2_object(fk2):
    fk2_fish = object_2_dict(fk2.fish)
    fk2_kite = object_2_dict(fk2.kite)

    assert fk2_fish == data_test_model3d.data_fk2["fk2_fish"]
    assert fk2_kite == data_test_model3d.data_fk2["fk2_kite"]


# %%
def test_df_creation(fk1, fk2):
    df1 = fk1.create_df()
    df2 = fk2.create_df()

    df1_ref_path = os.path.join(test_folder, "df_fk1_table.pkl")
    df2_ref_path = os.path.join(test_folder, "df_fk2_table.pkl")

    # export for reference
    # df1.to_pickle(df1_ref_path)
    # df2.to_pickle(df2_ref_path)

    df_ref1 = pd.read_pickle(df1_ref_path)
    df_ref2 = pd.read_pickle(df2_ref_path)

    pd.testing.assert_frame_equal(df1, df_ref1)
    pd.testing.assert_frame_equal(df2, df_ref2)
