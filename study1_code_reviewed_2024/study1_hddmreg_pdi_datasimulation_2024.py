#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:48:26 2024

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


#Data simulation check

data= hddm.load_csv('data_study1.csv')
data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

data_sim=pd.read_csv('study1_data_sim_pdigroups.csv')

colours={"estimated": "green",
         "simulated": "goldenrod"}

size_font=40

#%%

###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_mean_sim_quantiles
from quantile_simulation_function import get_quantiles


#get observed quantiles
data['coherence']=data['coherence'].astype('string')
observed_quantiles=get_quantiles(data)
#get simulated quantiles

#Create simulated data for parameter recovery
mreg_av_cond_PDI = hddm.load('hddm_lr_c_pdi')

mreg_av_cond_PDI.print_stats()
statsy=mreg_av_cond_PDI.gen_stats()
stats_s=statsy.T

mreg_av_cond_PDI_traces=mreg_av_cond_PDI.get_traces()



###drift rate
int_v=stats_s.at['mean','v_Intercept']
int_v_sd=stats_s.at['std','v_Intercept']
 
#PDI
b_v_pdi=stats_s.at['mean','v_z_PDI']*data.PDI.std() #destandardising beta

#condition
b_v_cond=stats_s.at['mean','v_coherence[T.25]']

###decision threshold
int_a=stats_s.at['mean','a_Intercept']

#PDI
b_a_pdi=stats_s.at['mean','a_z_PDI']

#condition
b_a_cond=stats_s.at['mean','a_coherence[T.25]']

###t
t=stats_s.at['mean','t']

#Generate data
trials_per_level = 15
subjs_per_bin=191


pdi=data.PDI



for x in pdi:
    xx = (pdi - pdi.mean()) / pdi.std()  # z-score the x factor
    
    a = int_a+b_a_pdi*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v = int_v+b_v_pdi*xx  # can also do for drift, here using same beta coeff
    
    a25 = int_a +b_a_cond+b_a_pdi*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v25 = int_v+b_v_cond+b_v_pdi*xx  # can also do for drift, here using same beta coeff
    
    
  
parvec_cond_0 = {'v':v.mean()  , 'a':a.mean()  , 't':t} 
parvec_cond_1 = {'v':v25.mean()  , 'a':a25.mean()  , 't':t} # set a to value set by regression, here v is set to constant


sample=range(500)
simulated_data = []


for i in sample:
    data_sim, params = hddm.generate.gen_rand_data({'parvec_cond_0':parvec_cond_0,
                                                 'parvec_cond_1':parvec_cond_1},
                                                      size = trials_per_level,
                                                      subjs=subjs_per_bin)
    
    data_sim.loc[data_sim['condition'] == 'parvec_cond_0', 'coherence'] = '15'
    data_sim.loc[data_sim['condition'] == 'parvec_cond_1', 'coherence'] = '25'

    
    data_sim=data_sim[['rt','response','coherence']]

    data_sim= data_sim[(data_sim['rt'] > 0.3)].assign(rt_type = "simulated")
    data_sim = data_sim[(data_sim['rt'] < 6)] 

    simulated_data.append(data_sim)
    pd.concat(simulated_data).to_csv('simulated_data_pdireg_2024.csv')
    
mean_simulated_quantiles_pdi=get_mean_sim_quantiles(simulated_data).assign(source = "simulated")

mean_simulated_quantiles_pdi.to_csv('mean_simulated_quantiles_pdi_reg_2024.csv')
#%%

simulated_data_pdireg=pd.read_csv('simulated_data_pdireg_2024.csv')

mean_simulated_quantiles_pdi=get_quantiles(simulated_data_pdireg)

mean_simulated_quantiles_pdi=mean_simulated_quantiles_pdi.assign(source = "simulated")
mean_simulated_quantiles_pdi.to_csv('mean_simulated_quantiles_pdi_reg_2024.csv')

observed_quantiles=observed_quantiles.assign(source = "observed")

#%%
# compute the mean squared error
from sklearn.metrics import mean_squared_error
mse='MSE:'

mse_1 = str(mean_squared_error(observed_quantiles['q_correct'], mean_simulated_quantiles_pdi['q_correct']).round(3))
mse_1="".join([mse,mse_1])
mse_0 = str(mean_squared_error(observed_quantiles['q_incorrect'], mean_simulated_quantiles_pdi['q_incorrect']).round(3))
mse_0="".join([mse,mse_0])
print(mse_1)
print(mse_0)

#qq plots
fig = make_subplots(
    rows=1, cols=2
)

fig.add_trace(go.Scatter(
    x=mean_simulated_quantiles_pdi['q_correct'],
    y=observed_quantiles['q_correct'],
    name='Correct'),
col=1, row=1)

fig.add_trace(go.Scatter(
    x=mean_simulated_quantiles_pdi['q_incorrect'],
    y=observed_quantiles['q_incorrect'],
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
                x1=observed_quantiles['q_correct'].max(),
                y1=observed_quantiles['q_correct'].max(),
                line=dict(color='black'),
                row=1,
                col=1)
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_incorrect'].max(),
                y1=observed_quantiles['q_incorrect'].max(),
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
    title='A. HDDM-LR-4 (Motion coherence - PDI)',    
    legend_title_font_color="blue",
    legend_font_size=size_font
)

fig.show()