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
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from fish_plot_3d import plot_3d_cases, plot_3d_cases_risingangle, plot_side_view
from model_3d import Project, FishKite
import alphashape

# %%
data_folder = os.path.join(os.path.dirname(__file__), "data")


# %%
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


# %% loop kite
list_result = []
for k_area in [10, 20, 25, 30, 40]:
    fk1.kite._flat_area = k_area
    dfi = fk1.create_df()
    dfvalid = dfi[dfi["isValid"]]
    lst_pt_i = dfvalid[["vmg_x_kt", "vmg_y_kt"]].to_numpy()
    alpha_shape_i = alphashape.alphashape(lst_pt_i, 0.2)
    list_result.append({"area": k_area, "shape": alpha_shape_i})

print("end loop")

# %% loop fish
list_result_fish = []
for k_area in [0.03, 0.06, 0.08, 0.1, 0.2]:
    print(f"compute {k_area}")
    fk1.fish._flat_area = k_area
    dfi = fk1.create_df()
    dfvalid = dfi[dfi["isValid"]]
    lst_pt_i = dfvalid[["vmg_x_kt", "vmg_y_kt"]].to_numpy()
    alpha_shape_i = alphashape.alphashape(lst_pt_i, 0.2)
    list_result_fish.append({"area": k_area, "shape": alpha_shape_i})

print("end loop fish")

# %%  plot

for r in list_result_fish:
    k_area = r["area"]
    ctn = r["shape"]
    x, y = ctn.exterior.xy
    area_score = ctn.area
    plt.plot(x, y, label=f"{k_area} m2 ({int(area_score)})")

# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
plt.rcParams["figure.figsize"] = [10, 10]
plt.axis("equal")
plt.grid()
plt.legend()
plt.xlabel("VMG_x_kt")
plt.ylabel("VMG_y_kt")
plt.title("Fish area sensibility")
plt.show()


# %% alpaha shape

lst_pt = df[["vmg_x_kt", "vmg_y_kt"]].to_numpy()


fig, ax = plt.subplots()
ax.scatter(*zip(*lst_pt))
plt.show()

# %%
alpha_shape = alphashape.alphashape(lst_pt, 0.2)
alpha_shape

# %%
fig, ax = plt.subplots()
ax.scatter(*zip(*lst_pt))
plt.plot(x, y, color="red")
# ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
plt.show()

# %%

x, y = alpha_shape.exterior.xy

# %%
