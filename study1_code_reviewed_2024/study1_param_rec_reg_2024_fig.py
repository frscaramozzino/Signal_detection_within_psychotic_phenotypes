#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:38:21 2023

@author: francescoscaramozzino
"""

import hddm
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from tableone import TableOne, load_dataset
import scipy.special as sps

import random
#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def calculate_dev(y_true,y_pred):
    return (2*(y_true * np.log(y_true/y_pred) - (y_true-y_pred))).sum()




mreg_av_cond_PDI = hddm.load('hddm_lr_c_pdi')
mreg_av_cond_CAPS = hddm.load('hddm_lr_c_caps')

sim_mreg_av_cond_PDI = hddm.load('sim_hddm_lr_c_pdi')
sim_mreg_av_cond_CAPS = hddm.load('sim_hddm_lr_c_caps')

mreg_av_cond_PDI_traces=mreg_av_cond_PDI.get_traces().assign(
    Source = "estimated", Schizotipy = "HDDM-LR-PDI-C")

mreg_av_cond_CAPS_traces=mreg_av_cond_CAPS.get_traces().assign(
    Source = "estimated", Schizotipy = "HDDM-LR-CAPS-C")

sim_mreg_av_cond_PDI_traces=sim_mreg_av_cond_PDI.get_traces().assign(
    Source = "simulated", Schizotipy = "HDDM-LR-PDI-C")
sim_mreg_av_cond_CAPS_traces=sim_mreg_av_cond_CAPS.get_traces().assign(
    Source = "simulated", Schizotipy = "HDDM-LR-CAPS-C")

# data for v plot
v_est_pdi= mreg_av_cond_PDI_traces[['v_z_PDI','v_coherence[T.25]','v_Intercept','Source','Schizotipy']]
v_est_pdi.rename(columns = {'v_z_PDI':'β Schizotipy',
                        'v_coherence[T.25]':'β Motion Coherence',
                        'v_Intercept':'Intercept'}, inplace = True)

v_est_caps= mreg_av_cond_CAPS_traces[['v_z_CAPS','v_coherence[T.25]','v_Intercept','Source','Schizotipy']]
v_est_caps.rename(columns = {'v_z_CAPS':'β Schizotipy',
                        'v_coherence[T.25]':'β Motion Coherence',
                        'v_Intercept':'Intercept'}, inplace = True)

v_est=pd.concat([v_est_pdi,v_est_caps])


v_sim_pdi= sim_mreg_av_cond_PDI_traces[['v_z_pdi','v_coherence[T.25]','v_Intercept','Source','Schizotipy']]
v_sim_pdi.rename(columns = {'v_z_pdi':'β Schizotipy',
                        'v_coherence[T.25]':'β Motion Coherence',
                        'v_Intercept':'Intercept'}, inplace = True)

v_sim_caps= sim_mreg_av_cond_CAPS_traces[['v_z_caps','v_coherence[T.25]','v_Intercept','Source','Schizotipy']]
v_sim_caps.rename(columns = {'v_z_caps':'β Schizotipy',
                        'v_coherence[T.25]':'β Motion Coherence',
                        'v_Intercept':'Intercept'}, inplace = True)


v_sim=pd.concat([v_sim_pdi,v_sim_caps])


v_comp=pd.concat([v_est,v_sim]).assign(Y = "Drift rate")

#data for plotiing a
a_est_pdi= mreg_av_cond_PDI_traces[['a_z_PDI','a_coherence[T.25]','a_Intercept','Source','Schizotipy']]
a_est_pdi.rename(columns = {'a_z_PDI':'β Schizotipy',
                        'a_coherence[T.25]':'β Motion Coherence',
                        'a_Intercept':'Intercept'}, inplace = True)

a_est_caps= mreg_av_cond_CAPS_traces[['a_z_CAPS','a_coherence[T.25]','a_Intercept','Source','Schizotipy']]
a_est_caps.rename(columns = {'a_z_CAPS':'β Schizotipy',
                        'a_coherence[T.25]':'β Motion Coherence',
                        'a_Intercept':'Intercept'}, inplace = True)


a_est=pd.concat([a_est_pdi,a_est_caps])


a_sim_pdi= sim_mreg_av_cond_PDI_traces[['a_z_pdi','a_coherence[T.25]','a_Intercept','Source','Schizotipy']]
a_sim_pdi.rename(columns = {'a_z_pdi':'β Schizotipy',
                        'a_coherence[T.25]':'β Motion Coherence',
                        'a_Intercept':'Intercept'}, inplace = True)

a_sim_caps= sim_mreg_av_cond_CAPS_traces[['a_z_caps','a_coherence[T.25]','a_Intercept','Source','Schizotipy']]
a_sim_caps.rename(columns = {'a_z_caps':'β Schizotipy',
                        'a_coherence[T.25]':'β Motion Coherence',
                        'a_Intercept':'Intercept'}, inplace = True)


a_sim=pd.concat([a_sim_pdi,a_sim_caps])


a_comp=pd.concat([a_est,a_sim]).assign(Y = "Decision threshold")

par_comp=pd.concat([v_comp,a_comp])
#%%
#######prob density plots

colours={"estimated": "gray",
         "simulated": "goldenrod"}
import numpy as np
from scipy.integrate import simps

# compute the mean squared error
from scipy.spatial import distance
from scipy.stats import gaussian_kde

jsd=distance.jensenshannon
tv=distance.cityblock

# Normalize distributions


def sum_to_1(dist1):
    norm_dist1 = dist1 / np.sum(dist1)
    return norm_dist1

def cohens_d(dist1, dist2):
    """Calculate Cohen's d between two distributions."""
    mean1 = np.mean(dist1)
    mean2 = np.mean(dist2)
    pooled_var = ((np.var(dist1) * (len(dist1) - 1)) + (np.var(dist2) * (len(dist2) - 1))) / (len(dist1) + len(dist2) - 2)
    effect_size = (mean1 - mean2) / np.sqrt(pooled_var)
    return effect_size


normal=stats.norm

csd='SMD:'
from sklearn.metrics import mean_squared_error


pdi_v_pdf=  (mreg_av_cond_PDI_traces['v_z_PDI'] )
sim_pdi_v_pdf=  (sim_mreg_av_cond_PDI_traces['v_z_pdi'] )
csd_pdi_v = str(cohens_d(pdi_v_pdf, sim_pdi_v_pdf).round(2))
csd_pdi_v="".join([csd,csd_pdi_v])

pdi_a_pdf=  (mreg_av_cond_PDI_traces['a_z_PDI'] )
sim_pdi_a_pdf=  (sim_mreg_av_cond_PDI_traces['a_z_pdi'] )
csd_pdi_a = str(cohens_d(pdi_a_pdf, sim_pdi_a_pdf).round(2))
csd_pdi_a="".join([csd,csd_pdi_a])


CAPS_v_pdf=  (mreg_av_cond_CAPS_traces['v_z_CAPS'] )
sim_CAPS_v_pdf=  (sim_mreg_av_cond_CAPS_traces['v_z_caps'] )
csd_CAPS_v = str(cohens_d(CAPS_v_pdf, sim_CAPS_v_pdf).round(2))
csd_CAPS_v="".join([csd,csd_CAPS_v])

CAPS_a_pdf=  (mreg_av_cond_CAPS_traces['a_z_CAPS'] )
sim_CAPS_a_pdf=  (sim_mreg_av_cond_CAPS_traces['a_z_caps'] )
csd_CAPS_a = str(cohens_d(CAPS_a_pdf, sim_CAPS_a_pdf).round(2))
csd_CAPS_a="".join([csd,csd_CAPS_a])



mse='RMSE:'

def compute_rmse_nrmse(est, sim):
    rmse = np.sqrt(mean_squared_error(est, sim))
    sd_true = np.std(est, ddof=1)
    nrmse = rmse / sd_true if sd_true != 0 else np.nan
    return f"RMSE:{rmse:.2f}", f"nRMSE:{nrmse:.2f}"

pdi_v_pdf = par_comp[
    (par_comp["Schizotipy"] == "HDDM-LR-PDI-C") &
    (par_comp["Y"] == "Drift rate") &
    (par_comp["Source"] == "estimated")
]


mse_pdi_v, nrmse_pdi_v = compute_rmse_nrmse(mreg_av_cond_PDI_traces['v_z_PDI'],sim_mreg_av_cond_PDI_traces['v_z_pdi'])
mse_pdi_a, nrmse_pdi_a = compute_rmse_nrmse(mreg_av_cond_PDI_traces['a_z_PDI'],sim_mreg_av_cond_PDI_traces['a_z_pdi'])
mse_CAPS_v, nrmse_CAPS_v = compute_rmse_nrmse(mreg_av_cond_CAPS_traces['v_z_CAPS'],sim_mreg_av_cond_CAPS_traces['v_z_caps'])
mse_CAPS_a, nrmse_CAPS_a = compute_rmse_nrmse(mreg_av_cond_CAPS_traces['a_z_CAPS'],sim_mreg_av_cond_CAPS_traces['a_z_caps'])




##### Beta Schizotipy
fig = px.histogram(par_comp, x="β Schizotipy", color='Source',  
                facet_col="Schizotipy",facet_row='Y',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,color_discrete_map=colours,   
                hover_data=par_comp.columns,
                barmode='overlay',
                width=1733,
                height=900)

# fig = px.violin(par_comp, y="v", x="Condition", box=True,  color='Source',  
#                 facet_col="Schizotipy",
#                 facet_row='Median_split',
#                  violinmode='overlay',
#                 hover_data=par_comp.columns,
#                 labels= {'v':'Drift rate'},
#                 width=1733,
#                 height=900)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.add_vline(x=0., line_width=1)


#credible interval
#v

def c_interval(dist1,dist2):
    q02_1=np.quantile(dist1, .05)
    q97_1=np.quantile(dist1, .95)
    q02_2=np.quantile(dist2, .05)
    q97_2=np.quantile(dist2, .95)
    return q02_1, q97_1,q02_2,q97_2
    
size_font=30
opac=0.6
###drift rate
#PDI
row=2
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['v_z_PDI'],
                    sim_mreg_av_cond_PDI_traces['v_z_pdi'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)
#credible interval
fig.add_annotation(
      y=30, x=mreg_av_cond_PDI_traces['v_z_PDI'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

fig.add_annotation(
      y=30, x=sim_mreg_av_cond_PDI_traces['v_z_pdi'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)

#RMSE
fig.add_annotation(
       y=30
     , x=0.15
     , text=mse_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=25
     , x=0.15
     , text=nrmse_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=35
     , x=0.15
     , text= csd_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)


#CAPS
row=2
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['v_z_CAPS'],
                    sim_mreg_av_cond_CAPS_traces['v_z_caps'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_annotation(
      y=30, x=mreg_av_cond_CAPS_traces['v_z_CAPS'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD
fig.add_annotation(
       y=30
     , x=0.15
     , text=mse_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=25
     , x=0.15
     , text=nrmse_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=35
     , x=0.15
     , text= csd_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

###decision threshold
#PDI
row=1
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['a_z_PDI'],
                    sim_mreg_av_cond_PDI_traces['a_z_pdi'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)

fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

fig.add_annotation(
      y=30, x=sim_mreg_av_cond_PDI_traces['a_z_pdi'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)

#JSD
fig.add_annotation(
       y=30
     , x=0.15
     , text=mse_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=25
     , x=0.15
     , text=nrmse_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)


#JSD
fig.add_annotation(
       y=35
     , x=0.15
     , text= csd_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
#CAPS
row=1
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['a_z_CAPS'],
                    sim_mreg_av_cond_CAPS_traces['a_z_caps'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_annotation(
      y=30, x=mreg_av_cond_CAPS_traces['a_z_CAPS'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

#JSD
fig.add_annotation(
       y=39
     , x=0.15
     , text=mse_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=25
     , x=0.15
     , text=nrmse_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=35
     , x=0.15
     , text= csd_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
        legendgrouptitle_text="Annotation",
    legendgroup="significant",
    name="Credible interval",
    mode="lines",
    line=dict(color="Black",dash='dot')
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P(β=0)<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=10)
))

fig.update_xaxes(tickfont=dict(size=28))
fig.update_yaxes(tickfont=dict(size=28),titlefont=dict(size=32))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_family="Times New Roman",
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=size_font
)
fig.update_xaxes(title_font_family="Arial")
fig.write_image("study1_par_rec_reg_2024.png",scale=3)

fig.show()
#%%

##### Beta condiiton


pdi_v_pdf=  (mreg_av_cond_PDI_traces['v_coherence[T.25]'])
sim_pdi_v_pdf=  (sim_mreg_av_cond_PDI_traces['v_coherence[T.25]'])
csd_pdi_v = str(  cohens_d(pdi_v_pdf, sim_pdi_v_pdf).round(2))
csd_pdi_v="".join([csd,csd_pdi_v])

pdi_a_pdf=  (mreg_av_cond_PDI_traces['a_coherence[T.25]'])
sim_pdi_a_pdf=  (sim_mreg_av_cond_PDI_traces['a_coherence[T.25]'])
csd_pdi_a = str(  cohens_d(pdi_a_pdf, sim_pdi_a_pdf).round(2))
csd_pdi_a="".join([csd,csd_pdi_a])


CAPS_v_pdf=  (mreg_av_cond_CAPS_traces['v_coherence[T.25]'])
sim_CAPS_v_pdf=  (sim_mreg_av_cond_CAPS_traces['v_coherence[T.25]'])
csd_CAPS_v = str(  cohens_d(CAPS_v_pdf, sim_CAPS_v_pdf).round(2))
csd_CAPS_v="".join([csd,csd_CAPS_v])

CAPS_a_pdf=  (mreg_av_cond_CAPS_traces['a_coherence[T.25]'])
sim_CAPS_a_pdf=  (sim_mreg_av_cond_CAPS_traces['a_coherence[T.25]'])
csd_CAPS_a = str(  cohens_d(CAPS_a_pdf, sim_CAPS_a_pdf).round(2))
csd_CAPS_a="".join([csd,csd_CAPS_a])


def compute_rmse_nrmse(est, sim):
    rmse = np.sqrt(mean_squared_error(est, sim))
    mean_true = np.mean(est)
    nrmse = rmse / abs(mean_true) if mean_true != 0 else np.nan
    return f"RMSE:{rmse:.2f}", f"nRMSE:{nrmse:.2f}"


mse_pdi_v, nrmse_pdi_v = compute_rmse_nrmse(mreg_av_cond_PDI_traces['v_coherence[T.25]'],sim_mreg_av_cond_PDI_traces['v_coherence[T.25]'])
mse_pdi_a, nrmse_pdi_a = compute_rmse_nrmse(mreg_av_cond_PDI_traces['a_coherence[T.25]'],sim_mreg_av_cond_PDI_traces['a_coherence[T.25]'])
mse_CAPS_v, nrmse_CAPS_v = compute_rmse_nrmse(mreg_av_cond_CAPS_traces['v_coherence[T.25]'],sim_mreg_av_cond_CAPS_traces['v_coherence[T.25]'])
mse_CAPS_a, nrmse_CAPS_a = compute_rmse_nrmse(mreg_av_cond_CAPS_traces['a_coherence[T.25]'],sim_mreg_av_cond_CAPS_traces['a_coherence[T.25]'])


# mse_pdi_v = str( mean_squared_error(mreg_av_cond_PDI_traces['v_coherence[T.25]'], sim_mreg_av_cond_PDI_traces['v_coherence[T.25]'],squared=False).round(3))
# mse_pdi_v="".join([mse,mse_pdi_v])
# mse_pdi_a = str( mean_squared_error(mreg_av_cond_PDI_traces['a_coherence[T.25]'], sim_mreg_av_cond_PDI_traces['a_coherence[T.25]'],squared=False).round(3))
# mse_pdi_a="".join([mse,mse_pdi_a])

# mse_CAPS_v = str( mean_squared_error(mreg_av_cond_CAPS_traces['v_coherence[T.25]'], sim_mreg_av_cond_CAPS_traces['v_coherence[T.25]'],squared=False).round(3))
# mse_CAPS_v="".join([mse,mse_CAPS_v])
# mse_CAPS_a = str( mean_squared_error(mreg_av_cond_CAPS_traces['a_coherence[T.25]'], sim_mreg_av_cond_CAPS_traces['a_coherence[T.25]'],squared=False).round(3))
# mse_CAPS_a="".join([mse,mse_CAPS_a])


fig = px.histogram(par_comp, x="β Motion Coherence", color='Source',  
                facet_col="Schizotipy",facet_row='Y',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,color_discrete_map=colours,   
                hover_data=par_comp.columns,
                barmode='overlay',
                width=1733,
                height=900)
fig.add_vline(x=0., line_width=1)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

###drift rate
#PDI
row=2
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['v_coherence[T.25]'],
                    sim_mreg_av_cond_PDI_traces['v_coherence[T.25]'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)
#credible interval
fig.add_annotation(
      y=15, x=mreg_av_cond_PDI_traces['v_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)
fig.add_annotation(
      y=15, x=sim_mreg_av_cond_PDI_traces['v_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD
fig.add_annotation(
       y=20
     , x=0.5
     , text=mse_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)



fig.add_annotation(
       y=23
     , x=0.5
     , text=csd_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

#CAPS
row=2
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['v_coherence[T.25]'],
                    sim_mreg_av_cond_CAPS_traces['v_coherence[T.25]'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_annotation(
      y=15, x=mreg_av_cond_CAPS_traces['v_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_annotation(
      y=15, x=sim_mreg_av_cond_CAPS_traces['v_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD
fig.add_annotation(
       y=20
     , x=0.5
     , text=mse_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=23
     , x=0.5
     , text=csd_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

###decision threshold
#PDI
row=1
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['a_coherence[T.25]'],
                    sim_mreg_av_cond_PDI_traces['a_coherence[T.25]'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

fig.add_annotation(
      y=15, x=mreg_av_cond_PDI_traces['a_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)
fig.add_annotation(
      y=15, x=sim_mreg_av_cond_PDI_traces['a_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)
#JSD
fig.add_annotation(
       y=20
     , x=0.5
     , text=mse_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=23
     , x=0.5
     , text=csd_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
#CAPS
row=1
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['a_coherence[T.25]'],
                    sim_mreg_av_cond_CAPS_traces['a_coherence[T.25]'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_annotation(
      y=15, x=mreg_av_cond_CAPS_traces['a_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['estimated'], family="Courier New, monospace"),
      row=row,col=col)
fig.add_annotation(
      y=15, x=sim_mreg_av_cond_CAPS_traces['a_coherence[T.25]'].mean(),
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color=colours['simulated'], family="Courier New, monospace"),
      row=row,col=col)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

#JSD
fig.add_annotation(
       y=20
     , x=0.5
     , text=mse_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=23
     , x=0.5
     , text=csd_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)



fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
        legendgrouptitle_text="Annotation",
    legendgroup="significant",
    name="Credible interval",
    mode="lines",
    line=dict(color="Black",dash='dot')
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P(β=0)<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=10)
))
fig.update_xaxes(tickfont=dict(size=28))
fig.update_yaxes(tickfont=dict(size=28),titlefont=dict(size=32))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_family="Times New Roman",
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=size_font
)
fig.update_xaxes(title_font_family="Arial")
fig.write_image("study1_par_rec_reg_motion2024.png",scale=3)

fig.show()
#%%
##### Intercept  v


pdi_v_pdf= (mreg_av_cond_PDI_traces['v_Intercept'])
sim_pdi_v_pdf= (sim_mreg_av_cond_PDI_traces['v_Intercept'])
csd_pdi_v = str(cohens_d(pdi_v_pdf, sim_pdi_v_pdf).round(2))
csd_pdi_v="".join([csd,csd_pdi_v])

pdi_a_pdf= (mreg_av_cond_PDI_traces['a_Intercept'])
sim_pdi_a_pdf= (sim_mreg_av_cond_PDI_traces['a_Intercept'])
csd_pdi_a = str(cohens_d(pdi_a_pdf, sim_pdi_a_pdf).round(2))
csd_pdi_a="".join([csd,csd_pdi_a])


CAPS_v_pdf= (mreg_av_cond_CAPS_traces['v_Intercept'])
sim_CAPS_v_pdf= (sim_mreg_av_cond_CAPS_traces['v_Intercept'])
csd_CAPS_v = str(cohens_d(CAPS_v_pdf, sim_CAPS_v_pdf).round(2))
csd_CAPS_v="".join([csd,csd_CAPS_v])

CAPS_a_pdf= (mreg_av_cond_CAPS_traces['a_Intercept'])
sim_CAPS_a_pdf= (sim_mreg_av_cond_CAPS_traces['a_Intercept'])
csd_CAPS_a = str(cohens_d(CAPS_a_pdf, sim_CAPS_a_pdf).round(2))
csd_CAPS_a="".join([csd,csd_CAPS_a])


ASI_v_pdf= (mreg_av_cond_ASI_traces['v_Intercept'])
sim_ASI_v_pdf= (sim_mreg_av_cond_ASI_traces['v_Intercept'])
csd_ASI_v = str(cohens_d(ASI_v_pdf, sim_ASI_v_pdf).round(2))
csd_ASI_v="".join([csd,csd_ASI_v])

ASI_a_pdf= (mreg_av_cond_ASI_traces['a_Intercept'])
sim_ASI_a_pdf= (sim_mreg_av_cond_ASI_traces['a_Intercept'])
csd_ASI_a = str(cohens_d(ASI_a_pdf, sim_ASI_a_pdf).round(2))
csd_ASI_a="".join([csd,csd_ASI_a])




mse_pdi_v = str(mean_squared_error(mreg_av_cond_PDI_traces['v_Intercept'], sim_mreg_av_cond_PDI_traces['v_Intercept'], squared=False).round(3))
mse_pdi_v="".join([mse,mse_pdi_v])
mse_pdi_a = str(mean_squared_error(mreg_av_cond_PDI_traces['a_Intercept'], sim_mreg_av_cond_PDI_traces['a_Intercept'], squared=False).round(3))
mse_pdi_a="".join([mse,mse_pdi_a])

mse_CAPS_v = str(mean_squared_error(mreg_av_cond_CAPS_traces['v_Intercept'], sim_mreg_av_cond_CAPS_traces['v_Intercept'], squared=False).round(3))
mse_CAPS_v="".join([mse,mse_CAPS_v])
mse_CAPS_a = str(mean_squared_error(mreg_av_cond_CAPS_traces['a_Intercept'], sim_mreg_av_cond_CAPS_traces['a_Intercept'], squared=False).round(3))
mse_CAPS_a="".join([mse,mse_CAPS_a])

mse_ASI_v = str(mean_squared_error(mreg_av_cond_ASI_traces['v_Intercept'], sim_mreg_av_cond_ASI_traces['v_Intercept'], squared=False).round(3))
mse_ASI_v="".join([mse,mse_ASI_v])
mse_ASI_a = str(mean_squared_error(mreg_av_cond_ASI_traces['a_Intercept'], sim_mreg_av_cond_ASI_traces['a_Intercept'], squared=False).round(3))
mse_ASI_a="".join([mse,mse_ASI_a])

fig = px.histogram(v_comp, x="Intercept", color='Source',  
                facet_col="Schizotipy",facet_row='Y',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,
                color_discrete_map=colours,                
                hover_data=par_comp.columns,
                barmode='overlay',
                width=1733,
                height=900)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

###drift rate
x_mse=0.9
y_mse=22.5
x_csd=0.9
y_csd=24.5
#PDI
row=1
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['v_Intercept'],
                    sim_mreg_av_cond_PDI_traces['v_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)
#credible interval


fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD

fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)


fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_pdi_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)


#CAPS
row=1
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['v_Intercept'],
                    sim_mreg_av_cond_CAPS_traces['v_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)



fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD
fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_CAPS_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
#ASI
row=1
col=3
a, b,c,d=c_interval(mreg_av_cond_ASI_traces['v_Intercept'],
                    sim_mreg_av_cond_ASI_traces['v_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)



#JSD
fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_ASI_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_ASI_v
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
        legendgrouptitle_text="Annotation",
    legendgroup="significant",
    name="Credible interval",
    mode="lines",
    line=dict(color="Black",dash='dot')
))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_family="Times New Roman",
    title='A.',
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=size_font
)
fig.update_xaxes(title_font_family="Arial")
fig.update_xaxes(tickfont=dict(size=28))
fig.update_yaxes(tickfont=dict(size=28),titlefont=dict(size=22))


fig.show()
#%%
##### Intercept 
fig = px.histogram(a_comp, x="Intercept", color='Source',  
                facet_col="Schizotipy",facet_row='Y',
                histnorm='probability density',
                orientation='v', 
                opacity=0.4,
                color_discrete_map=colours,
                hover_data=par_comp.columns,
                barmode='overlay',
                width=1733,
                height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
###decision threshold
x_mse=2.5
y_mse=20
x_csd=x_mse
y_csd=y_mse+1.5
#PDI
row=1
col=1
a, b,c,d=c_interval(mreg_av_cond_PDI_traces['a_Intercept'],
                    sim_mreg_av_cond_PDI_traces['a_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)


#JSD
fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_pdi_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
#CAPS
row=1
col=2
a, b,c,d=c_interval(mreg_av_cond_CAPS_traces['a_Intercept'],
                    sim_mreg_av_cond_CAPS_traces['a_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)



fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)

#JSD
fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_CAPS_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)
#ASI
row=1
col=3
a, b,c,d=c_interval(mreg_av_cond_ASI_traces['a_Intercept'],
                    sim_mreg_av_cond_ASI_traces['a_Intercept'])

fig.add_vline(x=a, line_dash="dot", row=row,col=col,
             line_color=colours['estimated'],
             opacity=opac)
fig.add_vline(x=b, line_dash="dot", row=row,col=col,
              line_color=colours['estimated'],
              opacity=opac)

fig.add_vline(x=c, line_dash="dot", row=row,col=col,
             line_color=colours['simulated'],
             opacity=opac)
fig.add_vline(x=d, line_dash="dot", row=row,col=col,
              line_color=colours['simulated'],
              opacity=opac)
#JSD
fig.add_annotation(
       y=y_mse
     , x=x_mse
     , text=mse_ASI_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)

fig.add_annotation(
       y=y_csd
     , x=x_csd
     , text=csd_ASI_a
     ,showarrow=False
     , font=dict(size=size_font, color="black", family="Courier New"),
    col=col, row=row)



fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_family="Times New Roman",
    title='B.',
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=size_font,
)
fig.update_xaxes(tickfont=dict(size=28))
fig.update_yaxes(tickfont=dict(size=28),titlefont=dict(size=22))
fig.update_xaxes(title_font_family="Arial")
fig.show()