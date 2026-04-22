#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:46:43 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:02:51 2023

@author: francescoscaramozzino
"""




import hddm
import pandas as pd
import scipy.stats as stats
import numpy as np

#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

from plotly.subplots import make_subplots
import plotly.graph_objects as go


data= hddm.load_csv('data_study1.csv')
data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)]

colours={"estimated": "gray",
         "simulated": "goldenrod"}

# grouping as suggested by Reviewer 1 as in Duhamel et al 2023

# Compute 81st percentile
caps_80 = np.percentile(data['caps'], 80)
print(caps_80)


# Drop rows where 0 < caps < caps_81
data = data.drop(data[(data['caps'] > 0) & (data['caps'] < caps_80)].index)

# Assign group labels
data['caps_group'] = np.where(data['caps'] == 0, 'low_caps', 'high_caps')

n_participants= len(data['sub_idx'].unique())
print(n_participants)


mean_simulated_quantiles_caps=pd.read_csv('mean_simulated_quantiles_caps_2025_reviewer1.csv')

#%%
###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_quantiles

#get quantiles
observed_quantiles=get_quantiles(data)

size_font=50

# compute the mean squared error
from sklearn.metrics import mean_squared_error
mse='RMSE:'

mse_1_caps = str(mean_squared_error(observed_quantiles['q_correct'], mean_simulated_quantiles_caps['q_correct'],squared=False).round(3))
mse_1_caps="".join([mse,mse_1_caps])
mse_0_caps = str(mean_squared_error(observed_quantiles['q_incorrect'], mean_simulated_quantiles_caps['q_incorrect'],squared=False).round(3))
mse_0_caps="".join([mse,mse_0_caps])

# mse_1_asi = str(mean_squared_error(observed_quantiles['q_correct'], mean_simulated_quantiles_asi['q_correct'],squared=False).round(3))
# mse_1_asi="".join([mse,mse_1_asi])
# mse_0_asi = str(mean_squared_error(observed_quantiles['q_incorrect'], mean_simulated_quantiles_asi['q_incorrect'],squared=False).round(3))
# mse_0_asi="".join([mse,mse_0_asi])

def cohens_d(dist1, dist2):
    """Calculate Cohen's d between two distributions."""
    mean1 = np.mean(dist1)
    mean2 = np.mean(dist2)
    pooled_var = ((np.var(dist1) * (len(dist1) - 1)) + (np.var(dist2)
                  * (len(dist2) - 1))) / (len(dist1) + len(dist2) - 2)
    effect_size = (mean1 - mean2) / np.sqrt(pooled_var)
    return effect_size
csd = 'SMD:'

csd_1_caps = str(cohens_d(observed_quantiles['q_correct'], mean_simulated_quantiles_caps['q_correct']).round(3))
csd_1_caps="".join([csd,csd_1_caps])
csd_0_caps = str(cohens_d(observed_quantiles['q_incorrect'], mean_simulated_quantiles_caps['q_incorrect']).round(3))
csd_0_caps="".join([csd,csd_0_caps])

# csd_1_asi = str(cohens_d(observed_quantiles['q_correct'], mean_simulated_quantiles_asi['q_correct']).round(3))
# csd_1_asi="".join([csd,csd_1_asi])
# csd_0_asi = str(cohens_d(observed_quantiles['q_incorrect'], mean_simulated_quantiles_asi['q_incorrect']).round(3))
# csd_0_asi="".join([csd,csd_0_asi])

#qq plots
fig = make_subplots(
    rows=1, cols=2,subplot_titles=("HDDM-CAPS-C-S", "HDDM-CAPS-C-S",
                                  ),
    horizontal_spacing=0.2,
    vertical_spacing=0.1)


#capsgroup

row=1
fig.add_trace(go.Scatter(
    x=observed_quantiles['q_correct'],
    y=mean_simulated_quantiles_caps['q_correct'],showlegend=True,mode='lines+markers',marker_color='blue',
    name='Correct'),
col=1, row=row)

fig.add_trace(go.Scatter(
    x=observed_quantiles['q_incorrect'],
    y=mean_simulated_quantiles_caps['q_incorrect'],showlegend=True,mode='lines+markers',marker_color='red',
    name='Incorrect'),
col=2, row=row)

fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_correct'].max(),
                y1=observed_quantiles['q_correct'].max(),
                line=dict(color='black'),opacity=0.25,
                row=row,
                col=1)
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_incorrect'].max(),
                y1=observed_quantiles['q_incorrect'].max(),
                line=dict(color='black'),opacity=0.25,
                row=row,
                col=2)

fig.add_annotation(
       y=3
     , x=1
     , text=mse_1_caps
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=1, row=row)
fig.add_annotation(
       y=3
     , x=1
     , text=mse_0_caps
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=2, row=row)

fig.add_annotation(
       y=3.5
     , x=1
     , text=csd_1_caps
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=1, row=row)
fig.add_annotation(
       y=3.5
     , x=1
     , text=csd_0_caps
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=2, row=row)


#asi
# row=2
# fig.add_trace(go.Scatter(
#     x=observed_quantiles['q_correct'],showlegend=False,mode='lines+markers',marker_color='blue',
#     y=mean_simulated_quantiles_asi['q_correct'],
#     name='Correct'),
# col=1, row=row)

# fig.add_trace(go.Scatter(
#     x=observed_quantiles['q_incorrect'],
#     y=mean_simulated_quantiles_asi['q_incorrect'],showlegend=False,mode='lines+markers',marker_color='red',
#     name='Incorrect'),
# col=2, row=row)

# fig.add_shape(type='line',
#                 x0=0,
#                 y0=0,
#                 x1=observed_quantiles['q_correct'].max(),
#                 y1=observed_quantiles['q_correct'].max(),
#                 line=dict(color='black'),opacity=0.25,
#                 row=row,
#                 col=1)
# fig.add_shape(type='line',
#                 x0=0,
#                 y0=0,
#                 x1=observed_quantiles['q_incorrect'].max(),
#                 y1=observed_quantiles['q_incorrect'].max(),
#                 line=dict(color='black'),opacity=0.25,
#                 row=row,
#                 col=2)

# fig.add_annotation(
#        y=3
#      , x=1
#      , text=mse_1_asi
#      ,showarrow=False
#      , font=dict(size=22, color="black", family="Courier New"),
#     col=1, row=row)
# fig.add_annotation(
#        y=3
#      , x=1
#      , text=mse_0_asi
#      ,showarrow=False
#      , font=dict(size=22, color="black", family="Courier New"),
#     col=2, row=row)

# fig.add_annotation(
#        y=3.5
#      , x=1
#      , text=csd_1_asi
#      ,showarrow=False
#      , font=dict(size=22, color="black", family="Courier New"),
#     col=1, row=row)
# fig.add_annotation(
#        y=3.5
#      , x=1
#      , text=csd_0_asi
#      ,showarrow=False
#      , font=dict(size=22, color="black", family="Courier New"),
#     col=2, row=row)


fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="Optimal simulation",
    mode="lines",opacity=0.25,
    line=dict(color="Black")
))

# Update xaxis properties
fig.update_xaxes(title_text="Observed RT quantiles", range=[0,4])
fig.update_annotations(font_size=size_font)

# Update yaxis properties
fig.update_yaxes(title_text="Simulated RT quantiles",range=[0,4])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_color="black",    
    legend_title_font_color="blue",
    legend_font_size=size_font,
    legend=dict(
    orientation="v",    
    entrywidth=500,
    yanchor="top",
    y=1.2,
    xanchor="left",
    x=0
),
    width=3200,
    height=2800,
)
fig.write_image("study1_predicted_quantiles_groups_2025_reviewer1.png",scale=3)

fig.show()
#%%