#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:05:44 2023

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:40:41 2023

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import statistics
import seaborn as sns

import numpy as np
#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#%%
#########
#Figures

#Get traces of parameters for plotting


mva_coh_pdi= hddm.load('va_PDI_25')
mva_coh_pdi_traces = mva_coh_pdi.get_traces()

mva_coh_caps= hddm.load('va_caps_25') 
mva_coh_caps_traces = mva_coh_caps.get_traces()

va_coh_asi= hddm.load('va_ASI_25') 
va_coh_asi_traces = va_coh_asi.get_traces()



###preparing dataset for plotting

#pdi
par_est_pdi=mva_coh_pdi_traces.assign(Schizotipy = "Delusion-like")


par_est_25_lp=par_est_pdi[['v(low_pdi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "Low", Psychometric= "Low PDI")
par_est_25_lp.rename(columns = {'v(low_pdi)':'v'}, inplace = True)


par_est_25_hp=par_est_pdi[['v(high_pdi)','t','Schizotipy']].assign( Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "High", Psychometric= "High PDI")
par_est_25_hp.rename(columns = {'v(high_pdi)':'v'}, inplace = True)


apar_est_25_lp=par_est_pdi[['a(low_pdi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "Low", Psychometric= "Low PDI")
apar_est_25_lp.rename(columns = {'a(low_pdi)':'a'}, inplace = True)


apar_est_25_hp=par_est_pdi[['a(high_pdi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "High", Psychometric= "High PDI")
apar_est_25_hp.rename(columns = {'a(high_pdi)':'a'}, inplace = True)


par_est_pdi=pd.concat([ par_est_25_lp,
                  par_est_25_hp,
                   apar_est_25_lp,
                  apar_est_25_hp,])

######
#caps

par_est_caps=mva_coh_caps_traces.assign(Schizotipy = "Hallucination-like")


par_est_25_lc=par_est_caps[['v(low_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
par_est_25_lc.rename(columns = {'v(low_caps)':'v'}, inplace = True)


par_est_25_hc=par_est_caps[['v(high_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
par_est_25_hc.rename(columns = {'v(high_caps)':'v'}, inplace = True)


apar_est_25_lc=par_est_caps[['a(low_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
apar_est_25_lc.rename(columns = {'a(low_caps)':'a'}, inplace = True)


apar_est_25_hc=par_est_caps[['a(high_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
apar_est_25_hc.rename(columns = {'a(high_caps)':'a'}, inplace = True)


par_est_caps=pd.concat([ par_est_25_lc,
                  par_est_25_hc,
                   apar_est_25_lc,
                 apar_est_25_hc])

######
#asi
par_est_asi=va_coh_asi_traces.assign( Schizotipy = "Aberrant salience")


par_est_25_la=par_est_asi[['v(low_asi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
par_est_25_la.rename(columns = {'v(low_asi)':'v'}, inplace = True)


par_est_25_ha=par_est_asi[['v(high_asi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
par_est_25_ha.rename(columns = {'v(high_asi)':'v'}, inplace = True)


apar_est_25_la=par_est_asi[['a(low_asi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
apar_est_25_la.rename(columns = {'a(low_asi)':'a'}, inplace = True)


apar_est_25_ha=par_est_asi[['a(high_asi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
apar_est_25_ha.rename(columns = {'a(high_asi)':'a'}, inplace = True)


par_est_asi=pd.concat( [par_est_25_la,
                  par_est_25_ha,
                   apar_est_25_la,
                  apar_est_25_ha])

par_caps_pdi=pd.concat([par_est_pdi, par_est_caps])

par_comp=pd.concat([par_caps_pdi, par_est_asi])


#all 
#%%
#CAPS only because winning model 

mva_coh_caps= hddm.load('hddm_caps') 
mva_coh_caps_traces = mva_coh_caps.get_traces()



######
#caps

par_est_caps=mva_coh_caps_traces.assign(Schizotipy = "Hallucination-like")


par_est_25_lc=par_est_caps[['v(low_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
     Median_split= "Low", Psychometric= "Low CAPS")
par_est_25_lc.rename(columns = {'v(low_caps)':'v'}, inplace = True)


par_est_25_hc=par_est_caps[['v(high_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
    Median_split= "High", Psychometric= "High CAPS")
par_est_25_hc.rename(columns = {'v(high_caps)':'v'}, inplace = True)


apar_est_25_lc=par_est_caps[['a(low_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
     Median_split= "Low", Psychometric= "Low CAPS")
apar_est_25_lc.rename(columns = {'a(low_caps)':'a'}, inplace = True)


apar_est_25_hc=par_est_caps[['a(high_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Median_split= "High", Psychometric= "High CAPS")
apar_est_25_hc.rename(columns = {'a(high_caps)':'a'}, inplace = True)


par_est_caps=pd.concat([ par_est_25_lc,
                  par_est_25_hc,
                   apar_est_25_lc,
                 apar_est_25_hc])



#%%


#######violin plot


colours={"Low PDI": "rgb(139, 224, 164)",
         "High PDI": "rgb(135, 197, 95)",
         "Low CAPS": "rgb(246, 207, 113)",
         "High CAPS": "rgb(248, 156, 116)",
         "Low ASI": "rgb(220, 176, 242)",
         "High ASI": "rgb(180, 151, 231)"}


#v

fig = px.violin(par_est_caps, y="v", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_row_spacing=0.05,
                color_discrete_map=colours, 
                labels= {'v':'Drift rate'},
                width=1500,
                height=900)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=40,
    title_font_family="Times New Roman",
    title_font_color="blue", 
    title='A.',
    title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    title_y=1,  # Adjust the vertical position of the title (0 to 1)
    legend_title_font_color="blue",
    legend_font_size=40,
    legend_title_text=None,
    legend=dict(
        orientation="v",    
        entrywidth=0,
        yanchor="bottom",
        y=0.5,
        xanchor="right",
        x=1.27
    )
)
fig.update_xaxes(title_font_family="Arial")

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

fig.add_annotation(
    x=0.005
    , y=1.1
    , text='<b>★<b>'
    ,showarrow=False
    , font=dict(size=40, color="black", family="Courier New, monospace")
    , xref='x1')

fig.show()

fig.write_image("study1_fig_v_prereg_caps_groups.png",scale=3)
#%%
#a
fig = px.violin(par_est_caps, y="a", box=True, color='Psychometric',  
                violinmode='overlay',
                color_discrete_map=colours,                
                labels= {'a':'Decision threshold'},
                width=1325,
                height=900)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Add interaction line



fig.update_traces(showlegend=False)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=40,
    title_font_family="Times New Roman",
    title="B.",
    title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    title_y=1,  # Adjust the vertical position of the title (0 to 1)
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=40
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x=0.005
    , y=2.52
    , text='<b>★<b>'
    ,showarrow=False
    , font=dict(size=40, color="black", family="Courier New, monospace")
    , xref='x1')


fig.show()
fig.write_image("study1_fig_a_prereg_caps_groups.png",scale=3)







#%%
#######violin plot


colours={"Low PDI": "rgb(139, 224, 164)",
         "High PDI": "rgb(135, 197, 95)",
         "Low CAPS": "rgb(246, 207, 113)",
         "High CAPS": "rgb(248, 156, 116)",
         "Low ASI": "rgb(220, 176, 242)",
         "High ASI": "rgb(180, 151, 231)"}


#v

fig = px.violin(par_comp, y="v", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_col="Schizotipy",facet_row_spacing=0.05,
                hover_data=par_comp.columns,
                color_discrete_map=colours, 
                labels= {'v':'Drift rate'},
                width=1500,
                height=900)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title_font_family="Times New Roman",
    title_font_color="blue", 
    title='A.  Motion coherence = 25%',
    legend_title_font_color="blue",
    legend_font_size=26,
    legend=dict(
        orientation="v",    
        entrywidth=0,
        yanchor="bottom",
        y=0.5,
        xanchor="right",
        x=1.2
    )
)
fig.update_xaxes(title_font_family="Arial")

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

fig.add_annotation(
    x=0.005
    , y=1.35
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x1')

# fig.add_annotation(
#     x=0.005
#     , y=1.35
#     , text='<b>*<b>'
#     ,showarrow=False
#     , font=dict(size=18, color="black", family="Courier New, monospace")
#     , xref='x2')

fig.add_annotation(
    x=0.005
    , y=1.35
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x3')
fig.show()

fig.write_image("study1_fig_v_prereg_25.png",scale=3)
#%%
#a
fig = px.violin(par_comp, y="a", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_col="Schizotipy",
                hover_data=par_comp.columns,
                color_discrete_map=colours,                
                labels= {'a':'Decision threshold'},
                width=1325,
                height=900)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Add interaction line



fig.update_traces(showlegend=False)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title_font_family="Times New Roman",
    title="B.  Motion coherence = 25%",
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x=0.005
    , y=2.7
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x2')


fig.show()
fig.write_image("study1_fig_a_prereg25.png",scale=3)
#%%
#####################################################


#########
#Figures 15

#Get traces of parameters for plotting


mva_coh_pdi= hddm.load('va_PDI_15')
mva_coh_pdi_traces = mva_coh_pdi.get_traces()

mva_coh_caps= hddm.load('va_caps_15') 
mva_coh_caps_traces = mva_coh_caps.get_traces()

va_coh_asi= hddm.load('va_ASI_15') 
va_coh_asi_traces = va_coh_asi.get_traces()



###preparing dataset for plotting

#pdi
par_est_pdi=mva_coh_pdi_traces.assign(Schizotipy = "Delusion-like")


par_est_25_lp=par_est_pdi[['v(low_pdi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "Low", Psychometric= "Low PDI")
par_est_25_lp.rename(columns = {'v(low_pdi)':'v'}, inplace = True)


par_est_25_hp=par_est_pdi[['v(high_pdi)','t','Schizotipy']].assign( Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "High", Psychometric= "High PDI")
par_est_25_hp.rename(columns = {'v(high_pdi)':'v'}, inplace = True)


apar_est_25_lp=par_est_pdi[['a(low_pdi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "Low", Psychometric= "Low PDI")
apar_est_25_lp.rename(columns = {'a(low_pdi)':'a'}, inplace = True)


apar_est_25_hp=par_est_pdi[['a(high_pdi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "High", Psychometric= "High PDI")
apar_est_25_hp.rename(columns = {'a(high_pdi)':'a'}, inplace = True)


par_est_pdi=pd.concat([ par_est_25_lp,
                  par_est_25_hp,
                   apar_est_25_lp,
                  apar_est_25_hp,])

######
#caps

par_est_caps=mva_coh_caps_traces.assign(Schizotipy = "Hallucination-like")


par_est_25_lc=par_est_caps[['v(low_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
par_est_25_lc.rename(columns = {'v(low_caps)':'v'}, inplace = True)


par_est_25_hc=par_est_caps[['v(high_caps)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
par_est_25_hc.rename(columns = {'v(high_caps)':'v'}, inplace = True)


apar_est_25_lc=par_est_caps[['a(low_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
apar_est_25_lc.rename(columns = {'a(low_caps)':'a'}, inplace = True)


apar_est_25_hc=par_est_caps[['a(high_caps)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
apar_est_25_hc.rename(columns = {'a(high_caps)':'a'}, inplace = True)


par_est_caps=pd.concat([ par_est_25_lc,
                  par_est_25_hc,
                   apar_est_25_lc,
                 apar_est_25_hc])

######
#asi
par_est_asi=va_coh_asi_traces.assign( Schizotipy = "Aberrant salience")


par_est_25_la=par_est_asi[['v(low_asi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
par_est_25_la.rename(columns = {'v(low_asi)':'v'}, inplace = True)


par_est_25_ha=par_est_asi[['v(high_asi)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
par_est_25_ha.rename(columns = {'v(high_asi)':'v'}, inplace = True)


apar_est_25_la=par_est_asi[['a(low_asi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
apar_est_25_la.rename(columns = {'a(low_asi)':'a'}, inplace = True)


apar_est_25_ha=par_est_asi[['a(high_asi)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
apar_est_25_ha.rename(columns = {'a(high_asi)':'a'}, inplace = True)


par_est_asi=pd.concat( [par_est_25_la,
                  par_est_25_ha,
                   apar_est_25_la,
                  apar_est_25_ha])

par_caps_pdi=pd.concat([par_est_pdi, par_est_caps])

par_comp=pd.concat([par_caps_pdi, par_est_asi])


#all 

#%%
#######violin plot


colours={"Low PDI": "rgb(139, 224, 164)",
         "High PDI": "rgb(135, 197, 95)",
         "Low CAPS": "rgb(246, 207, 113)",
         "High CAPS": "rgb(248, 156, 116)",
         "Low ASI": "rgb(220, 176, 242)",
         "High ASI": "rgb(180, 151, 231)"}


#v

fig = px.violin(par_comp, y="v", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_col="Schizotipy",facet_row_spacing=0.05,
                hover_data=par_comp.columns,
                color_discrete_map=colours, 
                labels= {'v':'Drift rate'},
                width=1500,
                height=900)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title_font_family="Times New Roman",
    title_font_color="blue", 
    title='C.  Motion coherence = 15%',
    legend_title_font_color="blue",
    legend_font_size=26,
    legend=dict(
        orientation="v",    
        entrywidth=0,
        yanchor="bottom",
        y=0.5,
        xanchor="right",
        x=1.2
    )
)
fig.update_xaxes(title_font_family="Arial")

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

fig.add_annotation(
    x=0.005
    , y=0.97
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x1')

fig.add_annotation(
    x=0.005
    , y=0.97
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x2')

fig.show()

fig.write_image("study1_fig_v_prereg_15.png",scale=3)
#%%
#a
fig = px.violin(par_comp, y="a", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_col="Schizotipy",
                hover_data=par_comp.columns,
                color_discrete_map=colours,                
                labels= {'a':'Decision threshold'},
                width=1325,
                height=900)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Add interaction line



fig.update_traces(showlegend=False)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title_font_family="Times New Roman",
    title="D.  Motion coherence = 15%",
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")


fig.show()
fig.write_image("study1_fig_a_prereg_15.png",scale=3)