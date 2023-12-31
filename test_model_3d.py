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
import jsonpickle
from pytest import fixture
from copy import deepcopy
import math
import numpy as np
import zipfile

cwd = os.getcwd()
test_folder = os.path.join(os.path.dirname(__file__), "test_assets")

test_data_folder = os.path.join(test_folder, "saved_data_for_test")

# %%"


# %% function
def object_2_dict(m_object, exclude=[]):
    attributes = dir(m_object)
    dict_data = {}

    # Iterate through the attributes and call only the callable ones (functions)
    for attr_name in attributes:
        if "__" in attr_name or attr_name in exclude:
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
        cable_strength_mm2=120,
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
def test_df_creation(fk1, fk2):
    overwrite_test_assets = False
    df1 = fk1.create_df()
    df2 = fk2.create_df()

    df1_filename = "df_fk1_table"
    df2_filename = "df_fk2_table"

    df1_ref_path = os.path.join(test_folder, f"{df1_filename}.pkl")
    df1_ref_path_zip = os.path.join(test_folder, f"{df1_filename}.zip")
    df2_ref_path = os.path.join(test_folder, f"{df2_filename}.pkl")
    # df1_ref_path_zip = os.path.join(test_folder, "df_fk1_table.zip")
    df2_ref_path_zip = os.path.join(test_folder, f"{df2_filename}.zip")

    # export for reference, this section should be commented for real test
    if overwrite_test_assets:
        print("/!\\ Test asset has been overwriten !! ")
        df1.to_pickle(df1_ref_path)
        with zipfile.ZipFile(df1_ref_path_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(df1_ref_path, arcname=f"{df1_filename}.pkl")

        # df2.to_pickle(df2_ref_path)
        df2.to_pickle(df2_ref_path)
        with zipfile.ZipFile(df2_ref_path_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(df2_ref_path, arcname=f"{df2_filename}.pkl")

    with zipfile.ZipFile(df1_ref_path_zip, mode="r") as archive:
        with archive.open(f"{df1_filename}.pkl") as pickle_file:
            df_ref1 = pd.read_pickle(pickle_file)
            pd.testing.assert_frame_equal(df1, df_ref1)

    with zipfile.ZipFile(df2_ref_path_zip, mode="r") as archive:
        with archive.open(f"{df2_filename}.pkl") as pickle_file:
            df_ref2 = pd.read_pickle(pickle_file)
            pd.testing.assert_frame_equal(df2, df_ref2)


# %%
def test_model_df_equality(fk1):
    """Check the function model are giving the same result than the table culculated

    Args:
        fk1 (_type_): _description_
    """
    # fk1_file = os.path.join(test_data_folder, "test_saved_fk1.json")
    # fk1 = FishKite.from_json(fk1_file, classes=FishKite)
    DEBUG = 0
    df_fk1 = fk1.create_df()

    list_col = df_fk1.columns
    attributes_fk1 = dir(fk1)

    common_column = [c for c in list_col if c in attributes_fk1]

    nb_row_to_check = 1000
    row_to_check = [int(r) for r in np.linspace(0, len(df_fk1) - 1, nb_row_to_check)]
    fk_test = deepcopy(fk1)
    i = 1
    for r_i in row_to_check:
        if DEBUG:
            print(f"{i} / {len(row_to_check)} :", end="")

        row_ref = df_fk1.iloc[r_i]

        fk_test.fish.cl = row_ref["fish_cl"]
        fk_test.kite.cl = row_ref["kite_cl"]
        fk_test.rising_angle = row_ref["rising_angle"]
        fk_test._extra_angle = row_ref["extra_angle"]

        if len(row_ref):
            for v in common_column:
                # if DEBUG:
                #     print(v)
                attr = getattr(fk_test, v)

                if callable(attr):
                    model_value = attr()

                else:
                    model_value = attr

                df_value = row_ref[v]
                assert math.isclose(
                    model_value, df_value
                ), f"fail at iloc: {r_i} for {v}"
            if DEBUG:
                print(f" OK")
        else:
            if DEBUG:
                print(f" X")
        i += 1


# %%"
