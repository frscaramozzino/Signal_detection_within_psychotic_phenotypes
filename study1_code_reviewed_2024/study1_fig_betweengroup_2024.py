#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:44:21 2024

@author: francescoscaramozzino
"""


import hddm
import pandas as pd


#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
import plotly.graph_objects as go

#########
#Figures

#Get traces of parameters for plotting


mva_coh_caps= hddm.load('hddm_c_caps') 
mva_coh_caps_traces = mva_coh_caps.get_traces()

va_coh_asi= hddm.load('hddm_c_asi') 
va_coh_asi_traces = va_coh_asi.get_traces()



###preparing dataset for plotting


######
#caps

par_est_caps=mva_coh_caps_traces.assign(Schizotipy = "Hallucination-like")

par_est_15_lc=par_est_caps[['v(low_caps.15)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "Low Coherence", Median_split= "Low", Psychometric= "Low CAPS")
par_est_15_lc.rename(columns = {'v(low_caps.15)':'v'}, inplace = True)


par_est_25_lc=par_est_caps[['v(low_caps.25)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
par_est_25_lc.rename(columns = {'v(low_caps.25)':'v'}, inplace = True)


par_est_15_hc=par_est_caps[['v(high_caps.15)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "Low Coherence", Median_split= "High", Psychometric= "High CAPS")
par_est_15_hc.rename(columns = {'v(high_caps.15)':'v'}, inplace = True)


par_est_25_hc=par_est_caps[['v(high_caps.25)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
par_est_25_hc.rename(columns = {'v(high_caps.25)':'v'}, inplace = True)



apar_est_15_lc=par_est_caps[['a(low_caps.15)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "Low Coherence", Median_split= "Low", Psychometric= "Low CAPS")
apar_est_15_lc.rename(columns = {'a(low_caps.15)':'a'}, inplace = True)


apar_est_25_lc=par_est_caps[['a(low_caps.25)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low CAPS")
apar_est_25_lc.rename(columns = {'a(low_caps.25)':'a'}, inplace = True)


apar_est_15_hc=par_est_caps[['a(high_caps.15)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "Low Coherence", Median_split= "High", Psychometric= "High CAPS")
apar_est_15_hc.rename(columns = {'a(high_caps.15)':'a'}, inplace = True)


apar_est_25_hc=par_est_caps[['a(high_caps.25)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High CAPS")
apar_est_25_hc.rename(columns = {'a(high_caps.25)':'a'}, inplace = True)


par_est_caps=pd.concat([par_est_15_lc, par_est_25_lc,
                  par_est_15_hc,par_est_25_hc,
                  apar_est_15_lc, apar_est_25_lc,
                  apar_est_15_hc,apar_est_25_hc])
#%%
######
#asi
par_est_asi=va_coh_asi_traces.assign( Schizotipy = "Aberrant salience")

par_est_15_la=par_est_asi[['v(low_asi.15)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "Low Coherence",Median_split= "Low", Psychometric= "Low ASI")
par_est_15_la.rename(columns = {'v(low_asi.15)':'v'}, inplace = True)


par_est_25_la=par_est_asi[['v(low_asi.25)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
par_est_25_la.rename(columns = {'v(low_asi.25)':'v'}, inplace = True)


par_est_15_ha=par_est_asi[['v(high_asi.15)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "Low Coherence", Median_split= "High", Psychometric= "High ASI")
par_est_15_ha.rename(columns = {'v(high_asi.15)':'v'}, inplace = True)


par_est_25_ha=par_est_asi[['v(high_asi.25)','t','Schizotipy']].assign(Parameter='Drift rate',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
par_est_25_ha.rename(columns = {'v(high_asi.25)':'v'}, inplace = True)


apar_est_15_la=par_est_asi[['a(low_asi.15)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "Low Coherence",Median_split= "Low", Psychometric= "Low ASI")
apar_est_15_la.rename(columns = {'a(low_asi.15)':'a'}, inplace = True)


apar_est_25_la=par_est_asi[['a(low_asi.25)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence", Median_split= "Low", Psychometric= "Low ASI")
apar_est_25_la.rename(columns = {'a(low_asi.25)':'a'}, inplace = True)


apar_est_15_ha=par_est_asi[['a(high_asi.15)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "Low Coherence", Median_split= "High", Psychometric= "High ASI")
apar_est_15_ha.rename(columns = {'a(high_asi.15)':'a'}, inplace = True)


apar_est_25_ha=par_est_asi[['a(high_asi.25)','t','Schizotipy']].assign(Parameter='Decision threshold',
    Condition = "High Coherence",Median_split= "High", Psychometric= "High ASI")
apar_est_25_ha.rename(columns = {'a(high_asi.25)':'a'}, inplace = True)


par_est_asi=pd.concat([par_est_15_la, par_est_25_la,
                  par_est_15_ha,par_est_25_ha,
                  apar_est_15_la, apar_est_25_la,
                  apar_est_15_ha,apar_est_25_ha])


par_comp=pd.concat([par_est_caps, par_est_asi])


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

fig = px.violin(par_comp, y="v", x="Condition", box=True, color='Psychometric',  
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
    title="A.",
    title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    title_y=0.85,  # Adjust the vertical position of the title (0 to 1)
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=26,
    legend=dict(
        orientation="h",    
        entrywidth=0,
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    )
)
fig.update_xaxes(title_font_family="Arial")

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgrouptitle_text="Annotation",
    legendgroup="significant",
    name="P<0.05",
    mode="lines",
    line=dict(color="Black")
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="P>0.05",
    mode="lines",
    line=dict(color="Black", dash='dash')
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

# Add interaction line
fig.add_shape(type='line',
                x0='Low Coherence',
                y0=mva_coh_caps_traces['v(high_caps.15)'].mean(),
                x1='High Coherence',
                y1=mva_coh_caps_traces['v(high_caps.25)'].mean(),
                line=dict(color=colours['High CAPS']),
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low Coherence',
                y0=mva_coh_caps_traces['v(low_caps.15)'].mean(),
                x1='High Coherence',
                y1=mva_coh_caps_traces['v(low_caps.25)'].mean(),
                line=dict(color=colours['Low CAPS']),
                row=1,
                col=1)


# Add interaction line
fig.add_shape(type='line',
                x0='Low Coherence',
                y0=va_coh_asi_traces['v(high_asi.15)'].mean(),
                x1='High Coherence',
                y1=va_coh_asi_traces['v(high_asi.25)'].mean(),
                line=dict(color=colours['High ASI']),
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low Coherence',
                y0=va_coh_asi_traces['v(low_asi.15)'].mean(),
                x1='High Coherence',
                y1=va_coh_asi_traces['v(low_asi.25)'].mean(),
                line=dict(color=colours['Low ASI']),
                row=1,
                col=2)
  
            
fig.add_annotation(
    x='Low Coherence'
    , y=0.98
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x1')
# fig.add_annotation(
#     x='High Coherence'
#     , y=1.35
#     , text='<b>*<b>'
#     ,showarrow=False
#     , font=dict(size=18, color="black", family="Courier New, monospace")
#     , xref='x1')

# fig.add_annotation(
#     x='Low Coherence'
#     , y=0.98
#     , text='<b>*<b>'
#     ,showarrow=False
#     , font=dict(size=18, color="black", family="Courier New, monospace")
#     , xref='x2')

fig.add_annotation(
    x='High Coherence'
    , y=1.32
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x2')
fig.show()

fig.write_image("study1_fig_v_2024.png",scale=3)
#%%
#a
fig = px.violin(par_comp, y="a", x="Condition", box=True, color='Psychometric',  
                violinmode='overlay',
                facet_col="Schizotipy",
                hover_data=par_comp.columns,
                color_discrete_map=colours,                
                labels= {'a':'Decision threshold'},
                width=1500,
                height=800)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


# Add interaction line
fig.add_shape(type='line',
                x0='Low Coherence',
                y0=mva_coh_caps_traces['a(high_caps.15)'].mean(),
                x1='High Coherence',
                y1=mva_coh_caps_traces['a(high_caps.25)'].mean(),
                line=dict(color=colours['High CAPS']),
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low Coherence',
                y0=mva_coh_caps_traces['a(low_caps.15)'].mean(),
                x1='High Coherence',
                y1=mva_coh_caps_traces['a(low_caps.25)'].mean(),
                line=dict(color=colours['Low CAPS']),
                row=1,
                col=1)


# Add interaction line
fig.add_shape(type='line',
                x0='Low Coherence',
                y0=va_coh_asi_traces['a(high_asi.15)'].mean(),
                x1='High Coherence',
                y1=va_coh_asi_traces['a(high_asi.25)'].mean(),
                line=dict(color=colours['High ASI']),
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low Coherence',
                y0=va_coh_asi_traces['a(low_asi.15)'].mean(),
                x1='High Coherence',
                y1=va_coh_asi_traces['a(low_asi.25)'].mean(),
                line=dict(color=colours['Low ASI']),
                row=1,
                col=2)

fig.update_traces(showlegend=False)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title_font_family="Times New Roman",
    title="B.",
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x='High Coherence'
    , y=2.7
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace")
    , xref='x1')


fig.show()
fig.write_image("study1_fig_a_2024.png",scale=3)

