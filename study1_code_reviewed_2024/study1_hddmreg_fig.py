#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:10:58 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 09:26:37 2022

@author: francescoscaramozzino
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:21:20 2022

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

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
from plotly.subplots import make_subplots
import plotly.graph_objects as go


mreg_av_cond_PDI = hddm.load('hddm_lr_c_pdi')
mreg_av_cond_CAPS = hddm.load('hddm_lr_c_caps')
mreg_av_cond_ASI = hddm.load('hddm_lr_c_asi')

mreg_av_cond_PDI.print_stats()
mreg_av_cond_CAPS.print_stats()
mreg_av_cond_ASI.print_stats()

mreg_av_cond_PDI_traces=mreg_av_cond_PDI.get_traces()
mreg_av_cond_PDI_traces=mreg_av_cond_PDI_traces.assign(Schizotipy = "Delusion-like", β="PDI")
mreg_av_cond_CAPS_traces=mreg_av_cond_CAPS.get_traces()
mreg_av_cond_CAPS_traces=mreg_av_cond_CAPS.get_traces().assign(Schizotipy = "Hallucination-like",β="CAPS")
# mreg_av_cond_ASI_traces=mreg_av_cond_ASI.get_traces()
# mreg_av_cond_ASI_traces=mreg_av_cond_ASI.get_traces().assign(Schizotipy = "Aberrant salience",β="ASI")


# Plot betas for v

v_pdi=mreg_av_cond_PDI_traces[['v_z_PDI','Schizotipy','β']]
v_pdi.rename(columns = {'v_z_PDI':'beta'}, inplace = True)
v_pdi=v_pdi.assign(y= "Drift rate")

v_caps=mreg_av_cond_CAPS_traces[['v_z_CAPS','Schizotipy','β']]
v_caps.rename(columns = {'v_z_CAPS':'beta'}, inplace = True)
v_caps=v_caps.assign(y= "Drift rate")

# v_asi=mreg_av_cond_ASI_traces[['v_z_ASI','Schizotipy','β']]
# v_asi.rename(columns = {'v_z_ASI':'beta'}, inplace = True)
# v_asi=v_asi.assign(y= "Drift rate")

a_pdi=mreg_av_cond_PDI_traces[['a_z_PDI','Schizotipy','β']]
a_pdi.rename(columns = {'a_z_PDI':'beta'}, inplace = True)
a_pdi=a_pdi.assign(y= "Decision threshold")

a_caps=mreg_av_cond_CAPS_traces[['a_z_CAPS','Schizotipy','β']]
a_caps.rename(columns = {'a_z_CAPS':'beta'}, inplace = True)
a_caps=a_caps.assign(y= "Decision threshold")

# a_asi=mreg_av_cond_ASI_traces[['a_z_ASI','Schizotipy','β']]
# a_asi.rename(columns = {'a_z_ASI':'beta'}, inplace = True)
# a_asi=a_asi.assign(y= "Decision threshold")

# stdv_pdi=mreg_av_cond_PDI_traces['v_z_PDI'].std()

# null_shape=np.shape(mreg_av_cond_PDI_traces['v_z_PDI'])
# x_axis=np.random.normal(loc=0, scale=stdv_pdi, size=null_shape)
                        
# null_dist = pd.DataFrame(x_axis)
# null_dist= null_dist.assign(y= "Drift rate",β="Null distribution")
# null_dist2 = pd.DataFrame(x_axis)
# null_dist2= null_dist2.assign(y= "Decision threshold",β="Null distribution")

# null_dist.rename(columns = {0:'beta'}, inplace = True)
# null_dist2.rename(columns = {0:'beta'}, inplace = True)


betas=pd.concat([v_pdi, v_caps,
                 a_pdi, a_caps])



# dic={'Null Distribution':x_axis, 'β PDI':v_pdi, 'β CAPS':v_caps,'β ASI':v_asi}
# df = pd.DataFrame(dic)

#%%
#######prob density plots
coloursdist={
         "PDI": "rgb(135, 197, 95)",
         "CAPS": "rgb(248, 156, 116)",
         "ASI": "rgb(180, 151, 231)"}
#####v

fig = px.histogram(betas, x="beta", color='β',  
                facet_row="y",
                facet_col="Schizotipy",
                color_discrete_map=coloursdist,
                facet_row_spacing=0.08,
                histnorm='probability density',
                orientation='v',
                barmode='group',
                # opacity=0.8,
                labels= {'beta':'β'},
                width=1600,
                height=1800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="Credible interval",
    mode="lines",
    line=dict(color="Black", dash='dot')
))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")

fig.add_vline(x=0., line_width=1)

####v
mv_pdi=mreg_av_cond_PDI_traces['v_z_PDI'].mean()
print(mv_pdi)
q02_vpdi=np.quantile(mreg_av_cond_PDI_traces['v_z_PDI'], .025)
q97_vpdi=np.quantile(mreg_av_cond_PDI_traces['v_z_PDI'], .975)

mv_caps=mreg_av_cond_CAPS_traces['v_z_CAPS'].mean()
print(mv_caps)
q02_vcaps=np.quantile(mreg_av_cond_CAPS_traces['v_z_CAPS'], .025)
q97_vcaps=np.quantile(mreg_av_cond_CAPS_traces['v_z_CAPS'], .975)

# mv_asi=mreg_av_cond_ASI_traces['v_z_ASI'].mean()
# print(mv_asi)
# q02_vasi=np.quantile(mreg_av_cond_ASI_traces['v_z_ASI'], .025)
# q97_vasi=np.quantile(mreg_av_cond_ASI_traces['v_z_ASI'], .975)

#mean
# fig.add_vline(x=mv_pdi, row=2,col=1,
#               annotation_text="0.07",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom right",
#               line_color='blue',
#               opacity=0.4)
#credible interval
fig.add_vline(x=q02_vpdi, line_dash="dot", row=2,col=1,
             line_color='rgb(135, 197, 95)',
             opacity=0.4)
fig.add_vline(x=q97_vpdi, line_dash="dot", row=2,col=1,
              line_color='rgb(135, 197, 95)',
              opacity=0.4)

fig.add_annotation(
      y=25, x=mv_pdi,
      text='<b>★<b>',
      showarrow=False,
      font=dict(size=45, color="black", family="Courier New, monospace"),
      row=2,col=1,)

#mean
# fig.add_vline(x=mv_caps, row=2,col=2,
#               annotation_text="0.08",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom right",
#               line_color='red',
#               opacity=0.4)
#credible interval
fig.add_vline(x=q02_vcaps, line_dash="dot", row=2,col=2,
              line_color='rgb(248, 156, 116)')


fig.add_vline(x=q97_vcaps, line_dash="dot", row=2,col=2,
              line_color='rgb(248, 156, 116)')



fig.add_annotation(
      y=25, x=mv_caps,
      text='<b>★<b>',
      showarrow=False,
      font=dict(size=45, color="black", family="Courier New, monospace"),
      row=2,col=2,)

#mean
# fig.add_vline(x=mv_asi, row=2, col=3,
#               annotation_text="0.008",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom right",
#               line_color='green',
#               opacity=0.4)   

           
#credible interval
# fig.add_vline(x=q02_vasi, line_dash="dot", row=2,col=3,
#               line_color='rgb(180, 151, 231)')


              
# fig.add_vline(x=q97_vasi, line_dash="dot", row=2,col=3,
#               line_color='rgb(180, 151, 231)')





####a
ma_pdi=mreg_av_cond_PDI_traces['a_z_PDI'].mean()
print(ma_pdi)
q02_apdi=np.quantile(mreg_av_cond_PDI_traces['a_z_PDI'], .025)
q97_apdi=np.quantile(mreg_av_cond_PDI_traces['a_z_PDI'], .975)

ma_caps=mreg_av_cond_CAPS_traces['a_z_CAPS'].mean()
print(ma_caps)
q02_acaps=np.quantile(mreg_av_cond_CAPS_traces['a_z_CAPS'], .025)
q97_acaps=np.quantile(mreg_av_cond_CAPS_traces['a_z_CAPS'], .975)

# ma_asi=mreg_av_cond_ASI_traces['a_z_ASI'].mean()
# print(ma_asi)
# q02_aasi=np.quantile(mreg_av_cond_ASI_traces['a_z_ASI'], .025)
# q97_aasi=np.quantile(mreg_av_cond_ASI_traces['a_z_ASI'], .975)


# fig.add_vline(x=ma_pdi, row=1,col=1,
#               annotation_text="-0.01",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom left",
#               line_color='blue',
#               opacity=0.4)
#credible interval
fig.add_vline(x=q02_apdi, line_dash="dot", row=1,col=1,
             line_color='rgb(135, 197, 95)')


fig.add_vline(x=q97_apdi, line_dash="dot", row=1,col=1,
              line_color='rgb(135, 197, 95)')



# fig.add_vline(x=ma_caps, row=1,col=2,
#               annotation_text="-0.04",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom left",
#               line_color='red',
#               opacity=0.4)

fig.add_annotation(
      y=25, x=ma_caps,
      text='<b>★<b>',
      showarrow=False,
      font=dict(size=45, color="black", family="Courier New, monospace"),
      row=1,col=2,)

#credible interval
fig.add_vline(x=q02_acaps, line_dash="dot", row=1,col=2,
              line_color='rgb(248, 156, 116)')


fig.add_vline(x=q97_acaps, line_dash="dot", row=1,col=2,
              line_color='rgb(248, 156, 116)')



# fig.add_vline(x=ma_asi,  row=1, col=3,
#               annotation_text="-0.01",
#               annotation_font_size=18,
#               annotation_font_color="black",
#               annotation_position="bottom left",
#               line_color='green',
#               opacity=0.4)
#credible interval
# fig.add_vline(x=q02_aasi, line_dash="dot", row=1,col=3,
#               line_color='rgb(180, 151, 231)')


              
# fig.add_vline(x=q97_aasi, line_dash="dot", row=1,col=3,
#               line_color='rgb(180, 151, 231)')


              

# fig.add_shape(type='rect', x0=-0.05, y0=0, x1=0.05, y1=30, 
#               line=dict(color="black",width=0.5,),
#               row=1,
#               col=1,
#               fillcolor='black',
#               opacity=0.2) 
              
# fig.add_shape(type='rect', x0=-0.05, y0=0, x1=0.05, y1=30, 
#               line=dict(color="black",width=0.5,),
#               row=2,
#               col=1,
#               fillcolor='black',
#               opacity=0.2) 
   
fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=40,
    title_font_family="Times New Roman",
    title_font_color="black",
     legend_title_text=None,
    #legend_title_font_color="blue",
    legend_font_size=40,
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=1.05,
        xanchor="left",
        x=0
    )
)
fig.show()
fig.write_image("study1_fig_hddm_reg_2024.png",scale=3)

