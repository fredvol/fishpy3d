# %% FISH Kite model 
import numpy as np
import pandas as pd


#%%

RHO_AIR = 1.29 # kg/m3
RHO_WATER = 1025 # kg/m3



wind_speed_i = 15 #kt
rising_angle_i= 33 #deg



class Deflector():
    # init method or constructor
    def __init__(self, name: str, cl :float,cl_range: tuple, area :float, efficiency_angle: float):
        self.name = name
        self.cl =  cl
        self.area = area #m2
        self.efficiency_angle =efficiency_angle  #deg

        self.cl_range = {'min':cl_range[0], 'max':cl_range[1]}

    # Sample Method
    def glide_ratio(self) -> float:
        return 1/np.tan(np.radians(self.efficiency_angle))
    
    def projected_efficiency_angle(self, m_raising_angle) -> float:
        return np.degrees(np.arctan(1 / (np.cos(np.radians(m_raising_angle)) * self.glide_ratio())))
    
class FishKite():
    def __init__(self, name: str, wind_speed :float, rising_angle :float , fish: Deflector ,kite : Deflector):

        self.name = name
        self.wind_speed =  wind_speed #kt
        self.rising_angle = rising_angle #deg
        self.fish =fish  
        self.kite =kite  

    def test(self):
        print('ok')

    def projected_efficiency_angle(self, what:str) -> float:
        if what=='kite':
            return self.kite.projected_efficiency_angle(self.rising_angle)
        elif what == 'fish':
            return self.fish.projected_efficiency_angle(self.rising_angle)
        else:
            print(f" {what} is unkown , waiting for 'fish' or 'kite' ")

    def total_efficiency(self):
        return self.kite.projected_efficiency_angle(self.rising_angle) + self.fish.projected_efficiency_angle(self.rising_angle)

    def fluid_velocity_ratio(self):
        current_ratio=  (RHO_AIR * self.kite.area * self.kite.cl / (RHO_WATER * self.fish.area * self.fish.cl)) ** 0.5
        return current_ratio
    
    def fluid_velocity_ratio_range(self):
        min_ratio =  (RHO_AIR * self.kite.area * self.kite.cl_range['min'] / (RHO_WATER * self.fish.area * self.fish.cl_range['max'])) ** 0.5
        max_ratio =  (RHO_AIR * self.kite.area * self.kite.cl_range['max'] / (RHO_WATER * self.fish.area * self.fish.cl_range['min'])) ** 0.5

        return {'max': max_ratio,'min': min_ratio }

    def true_wind_angle(self,velocity_ratio):
        #TODO to clean 
        value=np.degrees(np.arctan(np.sin(np.radians(self.total_efficiency()))/(velocity_ratio-np.cos(np.radians(self.total_efficiency())))))
        if value >0:
            return 180 - value
        else: 
            return  180- (180+value)
        
    def apparent_wind(self, velocity_ratio):
        apparent_wind_kt =self.wind_speed*np.sin(np.radians(180-self.true_wind_angle(velocity_ratio)))/np.sin(np.radians(self.total_efficiency()))
        return apparent_wind_kt
    
    def apparent_watter(self, velocity_ratio):
        apparent_water_kt =self.fluid_velocity_ratio() * self.apparent_wind(velocity_ratio)
        return apparent_water_kt
    
    def compute_polar(self, nb_value=21):
        velocity_max_min= self.fluid_velocity_ratio_range()
        velocity_range = np.linspace( velocity_max_min['min'] , velocity_max_min['max'] ,nb_value)
        print(list(velocity_range))

        list_result=[]
        for velocity_ratio in velocity_range:
            dict_i ={"velocity_ratio":velocity_ratio,
                     "true_wind_angle":self.true_wind_angle(velocity_ratio),
                     "apparent_wind_kt":self.apparent_wind(velocity_ratio),
                      }
            list_result.append(dict_i)

        df_polar = pd.DataFrame(list_result)

        df_polar['apparent_wind_pct'] = df_polar['apparent_wind_kt'] /self.wind_speed * 100 
        df_polar['apparent_watter_kt'] = df_polar['velocity_ratio'] *  df_polar['apparent_wind_kt']
        df_polar['apparent_watter_pct'] = df_polar['apparent_watter_kt'] /self.wind_speed * 100 
        df_polar['x_watter_pct'] = df_polar['apparent_watter_pct'] * np.sin(np.radians(df_polar['true_wind_angle']))
        df_polar['y_watter_pct'] = df_polar['apparent_watter_pct'] * np.cos(np.radians(df_polar['true_wind_angle']))

    
        return df_polar
    






d_kite =  Deflector('kite', cl=0.4,cl_range=(0.4,0.9) , area=24, efficiency_angle=12)
d_fish =  Deflector('fish', cl=0.2,cl_range=(0.2,0.4) , area=0.1,efficiency_angle= 14)

                          
fk = FishKite('fk1', wind_speed_i,rising_angle_i , fish= d_fish, kite=d_kite )
 
#%%
print(f"{d_kite.glide_ratio() =:.3f}")
print(f"{d_fish.glide_ratio() =:.3f}")
print(f"{d_kite.projected_efficiency_angle(43) =:.3f}")
print(f"{d_fish.projected_efficiency_angle(43) =:.3f}")
print(f"- fisk kite-")
print(f"{fk.projected_efficiency_angle('kite') =:.3f}")
print(f"{fk.total_efficiency() =:.3f}")
print("----")
#%%

print(f"{fk.fluid_velocity_ratio() =}")
vr=fk.fluid_velocity_ratio()
print(f"{fk.true_wind_angle(vr) =}")
print(f"{fk.apparent_wind(vr) =}")
print(f"{fk.apparent_watter(vr) =}")

df = fk.compute_polar()
df
# %%
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

fig = px.line_polar(df, r="apparent_watter_pct", theta="true_wind_angle", 
        range_theta=[-90,90],  
    
        )

# fig.update_polars(gridshape='linear')



# add wind
fig.add_trace(go.Scatterpolar(
        mode = "lines",
        r = [0,100],
        theta = [0,180],
        name='Wind_vector',
        line=dict(
            width =6,)
    
        )
    )

# trajectory 
vr=fk.fluid_velocity_ratio()
current_apparent_watter_pct = fk.apparent_watter(vr)/fk.wind_speed *100
current_true_wind_angle = fk.true_wind_angle(vr)


fig.add_trace(go.Scatterpolar(
        mode = "lines",
        r = [0,current_apparent_watter_pct],
        theta = [0,current_true_wind_angle],
        name='Traj_vector',
        )
    )


fig.add_trace(go.Scatterpolar(
        mode = "lines",
        r = [100,current_apparent_watter_pct],
        theta = [180,current_true_wind_angle],
        #legendgroup="vector",
        name='Apparent_wind_vector',
        line=dict(
            dash ='dot',)
        )
    )


# fig.update_polars( bgcolor="rgba(223, 223, 223, 0)")
# fig.add_layout_image(
#         dict(
#             source="polar_background.png",
#             xref="x",
#             yref="y",
#             x=0,
#             y=0,
#             sizex=4,
#             sizey=4,
#             sizing="stretch",
#             xanchor="left",
#             yanchor="middle",
#             opacity=0.7,
#             layer="below")
# )

fig.show()


#%% 
polar : L24, L25
