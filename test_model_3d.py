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
from model_3d import Deflector, FishKite, Pilot, load_fish_kite
import pandas as pd
import pickle
from pytest import fixture
import test_assets.data_test_model3d as data_test_model3d

cwd = os.getcwd()
test_folder = os.path.join(os.path.dirname(__file__), "test_assets")

test_data_folder = os.path.join(test_folder, "saved_data_for_test")

# %%"


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
    fk1_file = os.path.join(test_data_folder, "test_saved_fk1.json")
    # fk1 = load_fish_kite(fk1_file)
    fk1 = FishKite.from_json(fk1_file, classes=FishKite)

    return fk1


@fixture(scope="session")
def fk2():
    fk2_file = os.path.join(test_data_folder, "test_saved_fk2.json")
    fk2 = load_fish_kite(fk2_file)
    # fk2 = FishKite.from_json(fk2_file, classes=FishKite)
    return fk2


def test_save_load():
    """Test the save an load funstionallity by saving a file  , reloading it and comparing the difference"""
    fk_test_file = os.path.join(test_data_folder, "test_saved_fk_test.json")

    d_pilot2 = Pilot(mass=90, pilot_drag=0.35)
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

    fk_test = FishKite(
        "fk2",
        wind_speed=15,
        rising_angle=20,
        fish=d_fish2,
        kite=d_kite2,
        pilot=d_pilot2,
        extra_angle=20,
        cable_length_fish_unstreamline=28,
        cable_length_fish_streamline=2,
        cable_length_kite=12,
        cable_strength=500,
        cx_cable_water_streamline=0.4,
        cx_cable_water_unstreamline=1.1,
        cx_cable_air=1,
        tip_fish_depth=0.5,
    )

    fk_test.to_json(fk_test_file)

    # modify  first Fishkite
    fk_test.name = " original"

    # reload file

    fk_loaded = FishKite.from_json(fk_test_file)

    # check similarity

    assert fk_test.name != fk_loaded.name
    assert fk_test.wind_speed == fk_loaded.wind_speed

    print("deleted test file")
    os.remove(fk_test_file)


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
