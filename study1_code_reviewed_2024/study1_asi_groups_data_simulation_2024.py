#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:36:19 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:23:21 2024

@author: francescoscaramozzino
"""
""




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


#Data simulation check

data= hddm.load_csv('data_study1.csv')
data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

colours={"estimated": "green",
         "simulated": "goldenrod"}

size_font=22

#%%

###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_mean_sim_quantiles
from quantile_simulation_function import get_quantiles


#get observed quantiles
data['coherence']=data['coherence'].astype('string')
observed_quantiles_asi=get_quantiles(data)
#get simulated quantiles

#Create simulated data for parameter recovery

trials_per_level = 15
subjs_per_bin=96


mva_coh_asi= hddm.load('hddm_c_asi') 
mva_coh_asi.print_stats()
tms_s=mva_coh_asi.gen_stats()
tms_s=tms_s.T

#Set up parameters by conditions from previously run model
level_hp_15 = {'v':tms_s.at['mean','v(high_asi.15)'],
                 'a':tms_s.at['mean','a(high_asi.15)'], 
                 "t":tms_s.at['mean','t'], 
                 'sv':tms_s.at['std','v(high_asi.15)'], 
                 'sa':tms_s.at['std','a(high_asi.15)'], 
                 "st":tms_s.at['std','t']}


level_lp_15 = {'v':tms_s.at['mean','v(low_asi.15)'],
                 'a':tms_s.at['mean','a(low_asi.15)'], 
                 "t":tms_s.at['mean','t'], 
                 'sv':tms_s.at['std','v(low_asi.15)'], 
                 'sa':tms_s.at['std','a(low_asi.15)'], 
                 "st":tms_s.at['std','t']}

level_hp_25 = {'v':tms_s.at['mean','v(high_asi.25)'],
                 'a':tms_s.at['mean','a(high_asi.25)'], 
                 "t":tms_s.at['mean','t'], 
                 'sv':tms_s.at['std','v(high_asi.25)'], 
                 'sa':tms_s.at['std','a(high_asi.25)'], 
                 "st":tms_s.at['std','t']}

level_lp_25 = {'v':tms_s.at['mean','v(low_asi.25)'],
                 'a':tms_s.at['mean','a(low_asi.25)'], 
                 "t":tms_s.at['mean','t'], 
                 'sv':tms_s.at['std','v(low_asi.25)'], 
                 'sa':tms_s.at['std','a(low_asi.25)'], 
                 "st":tms_s.at['std','t']}


sample=range(500)
simulated_data = []


for i in sample:
    data_sim, params = hddm.generate.gen_rand_data({'level_hp_15':level_hp_15,
                                                 'level_lp_15':level_lp_15,
                                                 'level_hp_25':level_hp_25,
                                                 'level_lp_25':level_lp_25},
                                                      size = trials_per_level,
                                                      subjs=subjs_per_bin)
    
    data_sim.loc[data_sim['condition'] == 'level_hp_15', 'coherence'] = '15'
    data_sim.loc[data_sim['condition'] == 'level_lp_15', 'coherence'] = '15'
    data_sim.loc[data_sim['condition'] == 'level_hp_25', 'coherence'] = '25'
    data_sim.loc[data_sim['condition'] == 'level_lp_25', 'coherence'] = '25'
    
    data_sim.loc[data_sim['condition'] == 'level_hp_15', 'asi_group'] = 'high_asi'
    data_sim.loc[data_sim['condition'] == 'level_hp_25', 'asi_group'] = 'high_asi'
    data_sim.loc[data_sim['condition'] == 'level_lp_15', 'asi_group'] = 'low_asi'
    data_sim.loc[data_sim['condition'] == 'level_lp_25', 'asi_group'] = 'low_asi'
    
    data_sim=data_sim[['rt','response','coherence','asi_group']]

    data_sim= data_sim[(data_sim['rt'] > 0.3)].assign(rt_type = "simulated")
    data_sim = data_sim[(data_sim['rt'] < 6)] 

    simulated_data.append(data_sim)
    pd.concat(simulated_data).to_csv('simulated_data_asigroup_2024.csv')
    
mean_simulated_quantiles_asi=get_mean_sim_quantiles(simulated_data).assign(source = "simulated")

mean_simulated_quantiles_asi.to_csv('mean_simulated_quantiles_asi_2024.csv')
#%%

###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_mean_sim_quantiles
from quantile_simulation_function import get_condition_quantiles
from quantile_simulation_function import get_quantiles

simulated_data_asigroup=pd.read_csv('simulated_data_asigroup_2024.csv')

mean_simulated_quantiles_asi=get_quantiles(simulated_data_asigroup)
mean_simulated_quantiles_asi=mean_simulated_quantiles_asi.assign(source = "simulated")

mean_simulated_quantiles_asi.to_csv('mean_simulated_quantiles_asi_2024.csv')



#get observed quantiles
data['coherence']=data['coherence'].astype('string')
observed_quantiles_asi=get_condition_quantiles(data,'coherence','asi_group')
#get observed quantiles
observed_quantiles_asi=get_quantiles(data)
#get simulated quantiles
observed_quantiles_asi=observed_quantiles_asi.assign(source = "observed")

mean_simulated_quantiles_asi=mean_simulated_quantiles_asi.drop(columns=['quantile','source'])
observed_quantiles_asi=observed_quantiles_asi.drop(columns=['quantile','source'])



#%%
####plot actual and simulated rt
data_sim=data_sim[['rt','response','coherence','asi_group']]

data.loc[data['response'] == 1, 'response'] = 'Correct'
data.loc[data['response'] == 0, 'response'] = 'Incorrect'
data_sim.loc[data_sim['response'] == 1, 'response'] = 'Correct'
data_sim.loc[data_sim['response'] == 0, 'response'] = 'Incorrect'

rt_act=data[['rt','response','coherence','asi_group']]
rt_act=rt_act.assign(rt_type = "observed")

rt_sim=data_sim.assign(rt_type = "simulated")

comp_rt = pd.concat([rt_act, rt_sim])



fig = px.histogram(comp_rt, x="rt", color='rt_type',
                   histnorm='density',
                   facet_row='response',
                   orientation='v',
                   labels={'rt': 'Reaction time',
                            'response': 'Response',
                            'rt_type': 'Source'},
                   barmode='overlay',
                   color_discrete_map=colours,                   
                   width=1733,
                   height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Update yaxis properties

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font,
    title="RT Null-model",
    title_font_color="black",
    legend_title_font_color="blue",
    legend_font_size=size_font
)
fig.update_xaxes(title_font_family="Arial")

fig.show()
#%%
# compute the mean squared error
from sklearn.metrics import mean_squared_error
mse='MSE:'

mse_1 = str(mean_squared_error(observed_quantiles_asi['q_correct'], mean_simulated_quantiles_asi['q_correct']).round(3))
mse_1="".join([mse,mse_1])
mse_0 = str(mean_squared_error(observed_quantiles_asi['q_incorrect'], mean_simulated_quantiles_asi['q_incorrect']).round(3))
mse_0="".join([mse,mse_0])
print(mse_1)
print(mse_0)

#qq plots
fig = make_subplots(
    rows=1, cols=2
)

fig.add_trace(go.Scatter(
    x=mean_simulated_quantiles_asi['q_correct'],
    y=observed_quantiles_asi['q_correct'],
    name='Correct'),
col=1, row=1)

fig.add_trace(go.Scatter(
    x=mean_simulated_quantiles_asi['q_incorrect'],
    y=observed_quantiles_asi['q_incorrect'],
    name='Incorrect'),
col=2, row=1)

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="Optimal simulation",
    mode="lines",
    line=dict(color="Black")
))
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=mean_simulated_quantiles_asi['q_correct'].max(),
                y1=mean_simulated_quantiles_asi['q_correct'].max(),
                line=dict(color='black'),
                row=1,
                col=1)
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles_asi['q_incorrect'].max(),
                y1=observed_quantiles_asi['q_incorrect'].max(),
                line=dict(color='black'),
                row=1,
                col=2)

fig.add_annotation(
       y=5
     , x=3
     , text=mse_1
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=1, row=1)
fig.add_annotation(
       y=5
     , x=3
     , text=mse_0
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=2, row=1)

# Update xaxis properties
fig.update_xaxes(title_text="Observed RT quantiles", range=[0,6.5])

# Update yaxis properties
fig.update_yaxes(title_text="Simulated RT quantiles",range=[0,6.5])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_color="black",
    title='B. HDDM-5 (Motion coherence - CAPS groups)',        
    legend_title_font_color="blue",
    legend_font_size=size_font
)

fig.show()