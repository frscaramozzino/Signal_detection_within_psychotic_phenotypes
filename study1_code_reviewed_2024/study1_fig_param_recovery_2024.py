#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:25:12 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:25:27 2023

@author: francescoscaramozzino
"""


import hddm
import numpy as np
import pandas as pd
import scipy.stats as stats
#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


from plotly.subplots import make_subplots
import plotly.graph_objects as go


mva_coh_caps= hddm.load('hddm_c_caps') 
mva_coh_caps_traces = mva_coh_caps.get_traces()

vat_caps_coh_sim= hddm.load('hddm_c_caps_sim')
vat_caps_coh_sim_traces = vat_caps_coh_sim.get_traces()


va_coh_asi= hddm.load('hddm_c_asi') 
mva_coh_asi_traces = va_coh_asi.get_traces()

va_asi_coh_sim= hddm.load('hddm_c_asi_sim')
va_asi_coh_sim_traces = va_asi_coh_sim.get_traces()


#%%

a_L15, a_H15 = mva_coh_caps.nodes_db.node[['a(low_caps.15)', 'a(high_caps.15)']]
a_L25, a_H25 = mva_coh_caps.nodes_db.node[['a(low_caps.25)', 'a(high_caps.25)']]

s_a_L15, s_a_H15 = vat_caps_coh_sim.nodes_db.node[['a(low_caps.15)', 'a(high_caps.15)']]
s_a_L25, s_a_H25 = vat_caps_coh_sim.nodes_db.node[['a(low_caps.25)', 'a(high_caps.25)']]


print (("a15: sim_LCAPS > LCAPS="),(s_a_L15.trace()> a_L15.trace()).mean())
print (("a15: sim_HCAPS > vvHCAPS="),(s_a_H15.trace()> a_H15.trace()).mean())

print (("a25: sim_LCAPS > LCAPS="),(s_a_L25.trace()>a_L25.trace()).mean())
print (("a25: sim_HCAPS > HCAPS="),(s_a_H25.trace()>a_H25.trace()).mean())

#effect in simulated data
print (("Simulated effect a15: sim_LCAPS > sim_HCAPS="),(s_a_L15.trace()> s_a_H15.trace()).mean())
print (("Simulated effect a25: sim_CAPS > sim_HCAPS="),(s_a_L25.trace()> s_a_H25.trace()).mean())



#%%
###preparing dataset for plotting

######
#caps
par_est_caps=mva_coh_caps_traces.assign(Source = "estimated", Schizotipy = "Hallucination-like")

par_est_15_lcaps=par_est_caps[['v(low_caps.15)','a(low_caps.15)','t','Source','Schizotipy']].assign(Condition = "Low Precision", Median_split= "Low",Group= "Low CAPS")
par_est_15_lcaps.rename(columns = {'v(low_caps.15)':'v','a(low_caps.15)':'a'}, inplace = True)


par_est_25_lcaps=par_est_caps[['v(low_caps.25)','a(low_caps.25)','t','Source','Schizotipy']].assign(Condition = "High Precision",Median_split= "Low", Group= "Low CAPS")
par_est_25_lcaps.rename(columns = {'v(low_caps.25)':'v','a(low_caps.25)':'a'}, inplace = True)


par_est_15_hcaps=par_est_caps[['v(high_caps.15)','a(high_caps.15)','t','Source','Schizotipy']].assign(Condition = "Low Precision", Median_split= "High",Group= "High CAPS")
par_est_15_hcaps.rename(columns = {'v(high_caps.15)':'v','a(high_caps.15)':'a'}, inplace = True)


par_est_25_hcaps=par_est_caps[['v(high_caps.25)','a(high_caps.25)','t','Source','Schizotipy']].assign(Condition = "High Precision",Median_split= "High", Group= "High CAPS")
par_est_25_hcaps.rename(columns = {'v(high_caps.25)':'v','a(high_caps.25)':'a'}, inplace = True)


par_est_caps=pd.concat([par_est_15_lcaps, par_est_25_lcaps,
                  par_est_15_hcaps,par_est_25_hcaps])


par_sim_caps=vat_caps_coh_sim_traces.assign(Source = "simulated", Schizotipy = "Hallucination-like")

par_sim_15_lcaps=par_sim_caps[['v(low_caps.15)','a(low_caps.15)','t','Source','Schizotipy']].assign(Condition = "Low Precision", Median_split= "Low",Group= "Low CAPS")
par_sim_15_lcaps.rename(columns = {'v(low_caps.15)':'v','a(low_caps.15)':'a'}, inplace = True)


par_sim_25_lcaps=par_sim_caps[['v(low_caps.25)','a(low_caps.25)','t','Source','Schizotipy']].assign(Condition = "High Precision",Median_split= "Low", Group= "Low CAPS")
par_sim_25_lcaps.rename(columns = {'v(low_caps.25)':'v','a(low_caps.25)':'a'}, inplace = True)


par_sim_15_hcaps=par_sim_caps[['v(high_caps.15)','a(high_caps.15)','t','Source','Schizotipy']].assign(Condition = "Low Precision", Median_split= "High",Group= "High CAPS")
par_sim_15_hcaps.rename(columns = {'v(high_caps.15)':'v','a(high_caps.15)':'a'}, inplace = True)


par_sim_25_hcaps=par_sim_caps[['v(high_caps.25)','a(high_caps.25)','t','Source','Schizotipy']].assign(Condition = "High Precision",Median_split= "High", Group= "High CAPS")
par_sim_25_hcaps.rename(columns = {'v(high_caps.25)':'v','a(high_caps.25)':'a'}, inplace = True)


par_sim_caps=pd.concat([par_sim_15_lcaps, par_sim_25_lcaps,
                  par_sim_15_hcaps,par_sim_25_hcaps])


par_comp_caps=pd.concat([par_est_caps, par_sim_caps])


######
#asi
par_est_asi=mva_coh_asi_traces.assign(Source = "estimated", Schizotipy = "Aberrant salience")

par_est_15_lasi=par_est_asi[['v(low_asi.15)','a(low_asi.15)','t','Source','Schizotipy']].assign(
    Condition = "Low Coherence",Median_split= "Low", Group= "Low ASI")
par_est_15_lasi.rename(columns = {'v(low_asi.15)':'v','a(low_asi.15)':'a'}, inplace = True)


par_est_25_lasi=par_est_asi[['v(low_asi.25)','a(low_asi.25)','t','Source','Schizotipy']].assign(
    Condition = "High Coherence", Median_split= "Low", Group= "Low ASI")
par_est_25_lasi.rename(columns = {'v(low_asi.25)':'v','a(low_asi.25)':'a'}, inplace = True)


par_est_15_hasi=par_est_asi[['v(high_asi.15)','a(high_asi.15)','t','Source','Schizotipy']].assign(
    Condition = "Low Coherence", Median_split= "High", Group= "High ASI")
par_est_15_hasi.rename(columns = {'v(high_asi.15)':'v','a(high_asi.15)':'a'}, inplace = True)


par_est_25_hasi=par_est_asi[['v(high_asi.25)','a(high_asi.25)','t','Source','Schizotipy']].assign(
    Condition = "High Coherence",Median_split= "High", Group= "High ASI")
par_est_25_hasi.rename(columns = {'v(high_asi.25)':'v','a(high_asi.25)':'a'}, inplace = True)


par_est_asi=pd.concat([par_est_15_lasi, par_est_25_lasi,
                  par_est_15_hasi,par_est_25_hasi])


par_sim_asi=va_asi_coh_sim_traces.assign(Source = "simulated", Schizotipy = "Aberrant salience")

par_sim_15_lasi=par_sim_asi[['v(low_asi.15)','a(low_asi.15)','t','Source','Schizotipy']].assign(
    Condition = "Low Coherence",Median_split= "Low", Group= "Low ASI")
par_sim_15_lasi.rename(columns = {'v(low_asi.15)':'v','a(low_asi.15)':'a'}, inplace = True)


par_sim_25_lasi=par_sim_asi[['v(low_asi.25)','a(low_asi.25)','t','Source','Schizotipy']].assign(
    Condition = "High Coherence", Median_split= "Low", Group= "Low ASI")
par_sim_25_lasi.rename(columns = {'v(low_asi.25)':'v','a(low_asi.25)':'a'}, inplace = True)


par_sim_15_hasi=par_sim_asi[['v(high_asi.15)','a(high_asi.15)','t','Source','Schizotipy']].assign(
    Condition = "Low Coherence", Median_split= "High", Group= "High ASI")
par_sim_15_hasi.rename(columns = {'v(high_asi.15)':'v','a(high_asi.15)':'a'}, inplace = True)


par_sim_25_hasi=par_sim_asi[['v(high_asi.25)','a(high_asi.25)','t','Source','Schizotipy']].assign(
    Condition = "High Coherence",Median_split= "High", Group= "High ASI")
par_sim_25_hasi.rename(columns = {'v(high_asi.25)':'v','a(high_asi.25)':'a'}, inplace = True)


par_sim_asi=pd.concat([par_sim_15_lasi, par_sim_25_lasi,
                  par_sim_15_hasi,par_sim_25_hasi])


par_comp_asi=pd.concat([par_est_asi, par_sim_asi])


###final data for figures

par_comp=pd.concat([par_comp_caps, par_comp_asi])
par_sim_comp=pd.concat([ par_sim_caps, par_sim_asi])
#%%

import plotly.express as px
from scipy.stats import pearsonr

# Compute correlation
corr, pval = pearsonr(par_est_15_lcaps['a'], par_sim_15_lcaps['a'])
print(f"Pearson r = {corr:.3f}, p = {pval:.3g}")

# Create scatterplot
fig = px.scatter(
    x=par_est_15_lcaps['a'], 
    y=par_sim_15_lcaps['a'],
    labels={'x': 'Estimated a', 'y': 'Simulated a'},
    opacity=0.6,
    trendline="ols",   # adds regression line
    title=f"Pearson r = {corr:.3f}, p = {pval:.3g}"
)

# Add 1:1 reference line (y=x)
fig.add_shape(
    type="line",
    x0=min(par_est_15_lcaps['a']),
    y0=min(par_est_15_lcaps['a']),
    x1=max(par_est_15_lcaps['a']),
    y1=max(par_est_15_lcaps['a']),
    line=dict(color="red", dash="dash")
)

fig.show()
#%%

import matplotlib.pyplot as plt
import numpy as np

means = (par_est_15_lcaps['a'] + par_sim_15_lcaps['a']) / 2
diffs = par_est_15_lcaps['a'] - par_sim_15_lcaps['a']

plt.scatter(means, diffs, alpha=0.6)
plt.axhline(np.mean(diffs), color='red', linestyle='--')
plt.axhline(np.mean(diffs) + 1.96*np.std(diffs), color='gray', linestyle=':')
plt.axhline(np.mean(diffs) - 1.96*np.std(diffs), color='gray', linestyle=':')
plt.xlabel("Mean of Estimated & Simulated a")
plt.ylabel("Difference (Estimated - Simulated)")
plt.show()
#%%

means = (par_est_25_lcaps['a'] + par_sim_25_lcaps['a']) / 2
diffs = par_est_25_lcaps['a'] - par_sim_25_lcaps['a']

plt.scatter(means, diffs, alpha=0.6)
plt.axhline(np.mean(diffs), color='red', linestyle='--')
plt.axhline(np.mean(diffs) + 1.96*np.std(diffs), color='gray', linestyle=':')
plt.axhline(np.mean(diffs) - 1.96*np.std(diffs), color='gray', linestyle=':')
plt.xlabel("Mean of Estimated & Simulated a")
plt.ylabel("Difference (Estimated - Simulated)")
plt.show()
#%%

means = (par_est_15_hcaps['a'] + par_sim_15_hcaps['a']) / 2
diffs = par_est_15_hcaps['a'] - par_sim_15_hcaps['a']

plt.scatter(means, diffs, alpha=0.6)
plt.axhline(np.mean(diffs), color='red', linestyle='--')
plt.axhline(np.mean(diffs) + 1.96*np.std(diffs), color='gray', linestyle=':')
plt.axhline(np.mean(diffs) - 1.96*np.std(diffs), color='gray', linestyle=':')
plt.xlabel("Mean of Estimated & Simulated a")
plt.ylabel("Difference (Estimated - Simulated)")
plt.show()
#%%

means = (par_est_25_hcaps['a'] + par_sim_25_hcaps['a']) / 2
diffs = par_est_25_hcaps['a'] - par_sim_25_hcaps['a']

plt.scatter(means, diffs, alpha=0.6)
plt.axhline(np.mean(diffs), color='red', linestyle='--')
plt.axhline(np.mean(diffs) + 1.96*np.std(diffs), color='gray', linestyle=':')
plt.axhline(np.mean(diffs) - 1.96*np.std(diffs), color='gray', linestyle=':')
plt.xlabel("Mean of Estimated & Simulated a")
plt.ylabel("Difference (Estimated - Simulated)")
plt.show()
#%%
from scipy.stats import ttest_rel
t, p = ttest_rel(par_est_15_lcaps['a'], par_sim_15_lcaps['a'])
print(f"t = {t:.3f}, p = {p:.3g}")


#%%

size_font=40
# compute the mean squared error
from scipy.spatial import distance
jsd=distance.jensenshannon


def cohens_d(dist1, dist2):
    """Calculate Cohen's d between two distributions."""
    mean1 = np.mean(dist1)
    mean2 = np.mean(dist2)
    pooled_var = ((np.var(dist1) * (len(dist1) - 1)) + (np.var(dist2) * (len(dist2) - 1))) / (len(dist1) + len(dist2) - 2)
    effect_size = (mean1 - mean2) / np.sqrt(pooled_var)
    return effect_size

normal=stats.norm.pdf
from sklearn.metrics import mean_squared_error
mse='RMSE:'

csd='SMD:'
e='MSE:'

#%%

#CAPS groups
size_font=40
# compute the mean squared error
mse='RMSE:'

normal=stats.norm




def compute_rmse_nrmse(est, sim):
    rmse = np.sqrt(mean_squared_error(est, sim))
    mean_true = np.mean(est)
    nrmse = rmse / abs(mean_true) if mean_true != 0 else np.nan
    return f"RMSE:{rmse:.2f}", f"nRMSE:{nrmse:.2f}"


# --- Compute for each condition and parameter ---
mse_15_lcaps_v, nrmse_15_lcaps_v = compute_rmse_nrmse(par_est_15_lcaps['v'], par_sim_15_lcaps['v'])
mse_15_hcaps_v, nrmse_15_hcaps_v = compute_rmse_nrmse(par_est_15_hcaps['v'], par_sim_15_hcaps['v'])
mse_25_lcaps_v, nrmse_25_lcaps_v = compute_rmse_nrmse(par_est_25_lcaps['v'], par_sim_25_lcaps['v'])
mse_25_hcaps_v, nrmse_25_hcaps_v = compute_rmse_nrmse(par_est_25_hcaps['v'], par_sim_25_hcaps['v'])

mse_15_lcaps_a, nrmse_15_lcaps_a = compute_rmse_nrmse(par_est_15_lcaps['a'], par_sim_15_lcaps['a'])
mse_15_hcaps_a, nrmse_15_hcaps_a = compute_rmse_nrmse(par_est_15_hcaps['a'], par_sim_15_hcaps['a'])
mse_25_lcaps_a, nrmse_25_lcaps_a = compute_rmse_nrmse(par_est_25_lcaps['a'], par_sim_25_lcaps['a'])
mse_25_hcaps_a, nrmse_25_hcaps_a = compute_rmse_nrmse(par_est_25_hcaps['a'], par_sim_25_hcaps['a'])





#Cohen's d
caps_v15_lcaps_pdf= (par_est_15_lcaps['v'])
sim_caps_v15_lcaps_pdf= (par_sim_15_lcaps['v'])
csd_caps_v = str(  cohens_d(caps_v15_lcaps_pdf, sim_caps_v15_lcaps_pdf).round(2))
csd_15_lcaps_v="".join([csd,csd_caps_v])

caps_v25_lcaps_pdf= (par_est_25_lcaps['v'])
sim_caps_v25_lcaps_pdf= (par_sim_25_lcaps['v'])
csd_caps_v = str(  cohens_d(caps_v25_lcaps_pdf, sim_caps_v25_lcaps_pdf).round(2))
csd_25_lcaps_v="".join([csd,csd_caps_v])

caps_a15_lcaps_pdf= (par_est_15_lcaps['a'])
sim_caps_a15_lcaps_pdf= (par_sim_15_lcaps['a'])
csd_caps_a = str(  cohens_d(caps_a15_lcaps_pdf, sim_caps_a15_lcaps_pdf).round(2))
csd_15_lcaps_a="".join([csd,csd_caps_a])

caps_a25_lcaps_pdf= (par_est_25_lcaps['a'])
sim_caps_a25_lcaps_pdf= (par_sim_25_lcaps['a'])
csd_caps_a = str(  cohens_d(caps_a25_lcaps_pdf, sim_caps_a25_lcaps_pdf).round(2))
csd_25_lcaps_a="".join([csd,csd_caps_a])

caps_v15_hcaps_pdf= (par_est_15_hcaps['v'])
sim_caps_v15_hcaps_pdf= (par_sim_15_hcaps['v'])
csd_caps_v = str(cohens_d(caps_v15_hcaps_pdf, sim_caps_v15_hcaps_pdf).round(2) )
csd_15_hcaps_v="".join([csd,csd_caps_v])

caps_v25_hcaps_pdf= (par_est_25_hcaps['v'])
sim_caps_v25_hcaps_pdf= (par_sim_25_hcaps['v'])
csd_caps_v = str(cohens_d(caps_v25_hcaps_pdf, sim_caps_v25_hcaps_pdf).round(2) )
csd_25_hcaps_v="".join([csd,csd_caps_v])

caps_a15_hcaps_pdf= (par_est_15_hcaps['a'])
sim_caps_a15_hcaps_pdf= (par_sim_15_hcaps['a'])
csd_caps_a = str(cohens_d(caps_a15_hcaps_pdf, sim_caps_a15_hcaps_pdf).round(2) )
csd_15_hcaps_a="".join([csd,csd_caps_a])

caps_a25_hcaps_pdf= (par_est_25_hcaps['a'])
sim_caps_a25_hcaps_pdf= (par_sim_25_hcaps['a'])
csd_caps_a = str(cohens_d(caps_a25_hcaps_pdf, sim_caps_a25_hcaps_pdf).round(2) )
csd_25_hcaps_a="".join([csd,csd_caps_a])



#plot parameters
fig = make_subplots(
    rows=2, cols=2, column_titles=['Low Coherence', 'High Coherence'],vertical_spacing=0.1
)

colours={"estimated": "gray",
         "simulated": "goldenrod"}



# Drift Rate
#Low Precision
fig.add_trace(go.Violin(x=par_est_15_lcaps['Group'][ par_est_15_lcaps['Source'] == 'estimated'],
                        y=par_est_15_lcaps['v'][ par_est_15_lcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_sim_15_lcaps['Group'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        y=par_sim_15_lcaps['v'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',
                        line_color=colours['simulated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_est_15_hcaps['Group'][ par_est_15_hcaps['Source'] == 'estimated'],
                        y=par_est_15_hcaps['v'][ par_est_15_hcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_sim_15_hcaps['Group'][ par_sim_15_hcaps['Source'] == 'simulated' ],
                        y=par_sim_15_hcaps['v'][ par_sim_15_hcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=1)


fig.add_annotation(
       y=0.97
     , x='Low CAPS'
     , text=mse_15_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)
fig.add_annotation(
       y=1.07
     , x='High CAPS'
     , text=mse_15_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)




fig.add_annotation(
       y=0.92
     , x='Low CAPS'
     , text=nrmse_15_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)

fig.add_annotation(
       y=1.02
     , x='High CAPS'
     , text=nrmse_15_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)



fig.add_annotation(
       y=1.02
     , x='Low CAPS'
     , text=csd_15_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)

fig.add_annotation(
       y=1.12
     , x='High CAPS'
     , text=csd_15_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)

#High Precision

fig.add_trace(go.Violin(x=par_est_25_lcaps['Group'][ par_est_25_lcaps['Source'] == 'estimated'],
                        y=par_est_25_lcaps['v'][ par_est_25_lcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_sim_25_lcaps['Group'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        y=par_sim_25_lcaps['v'][ par_sim_25_lcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_est_25_hcaps['Group'][ par_est_25_hcaps['Source'] == 'estimated'],
                        y=par_est_25_hcaps['v'][ par_est_25_hcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", showlegend=False,
                        line_color=colours['estimated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_sim_25_hcaps['Group'][ par_sim_25_hcaps['Source'] == 'simulated' ],
                        y=par_sim_25_hcaps['v'][ par_sim_25_hcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=1)

fig.add_annotation(
       y=1.43
     , x='Low CAPS'
     , text=mse_25_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)
fig.add_annotation(
       y=1.4
     , x='High CAPS'
     , text=mse_25_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

fig.add_annotation(
       y=1.38
     , x='Low CAPS'
     , text=nrmse_25_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

fig.add_annotation(
       y=1.35
     , x='High CAPS'
     , text=nrmse_25_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

fig.add_annotation(
       y=1.48
     , x='Low CAPS'
     , text=csd_25_lcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

fig.add_annotation(
       y=1.45
     , x='High CAPS'
     , text=csd_25_hcaps_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)


#Decision theshold
#Low Precision
fig.add_trace(go.Violin(x=par_est_15_lcaps['Group'][ par_est_15_lcaps['Source'] == 'estimated'],
                        y=par_est_15_lcaps['a'][ par_est_15_lcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_sim_15_lcaps['Group'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        y=par_sim_15_lcaps['a'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_est_15_hcaps['Group'][ par_est_15_hcaps['Source'] == 'estimated'],
                        y=par_est_15_hcaps['a'][ par_est_15_hcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_sim_15_hcaps['Group'][ par_sim_15_hcaps['Source'] == 'simulated' ],
                        y=par_sim_15_hcaps['a'][ par_sim_15_hcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=2)


fig.add_annotation(
       y=2.53
     , x='Low CAPS'
     , text=mse_15_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)
fig.add_annotation(
       y=2.53
     , x='High CAPS'
     , text=mse_15_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)



fig.add_annotation(
       y=2.48
     , x='Low CAPS'
     , text=nrmse_15_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)

fig.add_annotation(
       y=2.48
     , x='High CAPS'
     , text=nrmse_15_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)


fig.add_annotation(
       y=2.58
     , x='Low CAPS'
     , text=csd_15_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)
fig.add_annotation(
       y=2.58
     , x='High CAPS'
     , text=csd_15_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)

#High Precision

fig.add_trace(go.Violin(x=par_est_25_lcaps['Group'][ par_est_25_lcaps['Source'] == 'estimated'],
                        y=par_est_25_lcaps['a'][ par_est_25_lcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_sim_25_lcaps['Group'][ par_sim_15_lcaps['Source'] == 'simulated' ],
                        y=par_sim_25_lcaps['a'][ par_sim_25_lcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_est_25_hcaps['Group'][ par_est_25_hcaps['Source'] == 'estimated'],
                        y=par_est_25_hcaps['a'][ par_est_25_hcaps['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", showlegend=False,
                        line_color=colours['estimated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_sim_25_hcaps['Group'][ par_sim_25_hcaps['Source'] == 'simulated' ],
                        y=par_sim_25_hcaps['a'][ par_sim_25_hcaps['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=2)

fig.add_annotation(
       y=2.99
     , x='Low CAPS'
     , text=mse_25_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)
fig.add_annotation(
       y=2.82
     , x='High CAPS'
     , text=mse_25_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)



fig.add_annotation(
       y=2.92
     , x='Low CAPS'
     , text=nrmse_25_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)
fig.add_annotation(
       y=2.75
     , x='High CAPS'
     , text=nrmse_25_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)


fig.add_annotation(
       y=3.06
     , x='Low CAPS'
     , text=csd_25_lcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)

fig.add_annotation(
       y=2.89
     , x='High CAPS'
     , text=csd_25_hcaps_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)




fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')
fig.update_annotations(font_size=50)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

opac=0.6


#add line to show interaction
#CAPS
fig.add_shape(type='line',
                x0='Low CAPS',
                y0=mva_coh_caps_traces['v(low_caps.15)'].mean(),
                x1='High CAPS',
                y1=mva_coh_caps_traces['v(high_caps.15)'].mean(),
                line=dict(color=colours['estimated']),opacity=opac,
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_caps_coh_sim_traces['v(low_caps.15)'].mean(),
                x1='High CAPS',
                y1=vat_caps_coh_sim_traces['v(high_caps.15)'].mean(),
                line=dict(color=colours['simulated']),opacity=opac,
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=mva_coh_caps_traces['v(low_caps.25)'].mean(),
                x1='High CAPS',
                y1=mva_coh_caps_traces['v(high_caps.25)'].mean(),
                line=dict(color=colours['estimated'], dash='dash'),opacity=opac,
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_caps_coh_sim_traces['v(low_caps.25)'].mean(),
                x1='High CAPS',
                y1=vat_caps_coh_sim_traces['v(high_caps.25)'].mean(),
                line=dict(color=colours['simulated'], dash='dash'),opacity=opac,
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=mva_coh_caps_traces['a(low_caps.15)'].mean(),
                x1='High CAPS',
                y1=mva_coh_caps_traces['a(high_caps.15)'].mean(),
                line=dict(color=colours['estimated'], dash='dash'),opacity=opac,
                row=2,
                col=1)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_caps_coh_sim_traces['a(low_caps.15)'].mean(),
                x1='High CAPS',
                y1=vat_caps_coh_sim_traces['a(high_caps.15)'].mean(),
                line=dict(color=colours['simulated'], dash='dash'),opacity=opac,
                row=2,
                col=1)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=mva_coh_caps_traces['a(low_caps.25)'].mean(),
                x1='High CAPS',
                y1=mva_coh_caps_traces['a(high_caps.25)'].mean(),
                line=dict(color=colours['estimated']),opacity=opac,
                row=2,
                col=2)

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_caps_coh_sim_traces['a(low_caps.25)'].mean(),
                x1='High CAPS',
                y1=vat_caps_coh_sim_traces['a(high_caps.25)'].mean(),
                line=dict(color=colours['simulated']),opacity=opac,
                row=2,
                col=2)


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


# Update yaxis properties
fig.update_yaxes(title_text="Drift rate", col=1, row=1)
fig.update_yaxes(title_text="Drift rate", col=2, row=1)
fig.update_yaxes(title_text="Decision threshold", col=1, row=2)
fig.update_yaxes(title_text="Decision threshold", col=2, row=2)


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font,
    title_font_color="blue", 
    legend_title_font_color="blue",
    legend_font_size=size_font,
    autosize=False,
    width=2800,
    height=2000,
)
fig.update_xaxes(title_font_family="Arial")
fig.write_image("study1_fig_caps_group_recovery_2024.png",scale=3)

fig.show()
#%%

mse_t = str(mean_squared_error(mva_coh_caps_traces[['t']], vat_caps_coh_sim_traces[['t']]).round(3))
mse_t="".join([mse,mse_t])

t_est=mva_coh_caps_traces[['t']].assign(Condition = "Fixed parameter",Source= "estimated")
t_sim=vat_caps_coh_sim_traces[['t']].assign(Condition = "Fixed parameter",Source= "simulated")

colours={"estimated": "gray",
         "simulated": "goldenrod"}

fig = make_subplots(
    rows=1, cols=1, 
)

fig.add_trace(go.Violin(x=t_est['Condition'][ t_est['Source'] == 'estimated' ],
                        y=t_est['t'][ t_est['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                        line_color=colours['estimated']), col=1, row=1)

fig.add_trace(go.Violin(x=t_sim['Condition'][ t_sim['Source'] == 'simulated' ],
                        y=t_sim['t'][ t_sim['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=1)

fig.add_annotation(
       y=0.375
     , x='Fixed parameter'
     , text=mse_t
     ,showarrow=False
     , font=dict(size=40, color="black", family="Courier New"),
    col=1, row=1)


fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# Update yaxis properties

fig.update_yaxes(title_text="Non-decision time",col=1, row=1)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font,
    title_font_color="black", 
    legend_title_font_color="blue",
    legend_font_size=size_font,
    autosize=False,
    width=1000,
    height=1000,
)
fig.update_xaxes(title_font_family="Arial")
fig.show()

#%%

#ASI groups
size_font=40
# compute the mean squared error




# asi_v15_lasi_pdf=normal.pdf(par_est_15_lasi['v'])
# sim_asi_v15_lasi_pdf=normal.pdf(par_sim_15_lasi['v'])
# mse_asi_v = str(jsd(asi_v15_lasi_pdf, sim_asi_v15_lasi_pdf).round(4))
# mse_15_lasi_v="".join([mse,mse_asi_v])

# asi_v25_lasi_pdf=normal.pdf(par_est_25_lasi['v'])
# sim_asi_v25_lasi_pdf=normal.pdf(par_sim_25_lasi['v'])
# mse_asi_v = str(jsd(asi_v25_lasi_pdf, sim_asi_v25_lasi_pdf).round(4))
# mse_25_lasi_v="".join([mse,mse_asi_v])

# asi_a15_lasi_pdf=normal.pdf(par_est_15_lasi['a'])
# sim_asi_a15_lasi_pdf=normal.pdf(par_sim_15_lasi['a'])
# mse_asi_a = str(jsd(asi_a15_lasi_pdf, sim_asi_a15_lasi_pdf).round(4))
# mse_15_lasi_a="".join([mse,mse_asi_a])

# asi_a25_lasi_pdf=normal.pdf(par_est_25_lasi['a'])
# sim_asi_a25_lasi_pdf=normal.pdf(par_sim_25_lasi['a'])
# mse_asi_a = str(jsd(asi_a25_lasi_pdf, sim_asi_a25_lasi_pdf).round(4))
# mse_25_lasi_a="".join([mse,mse_asi_a])


# asi_v15_hasi_pdf=normal.pdf(par_est_15_hasi['v'])
# sim_asi_v15_hasi_pdf=normal.pdf(par_sim_15_hasi['v'])
# mse_asi_v = str(jsd(asi_v15_hasi_pdf, sim_asi_v15_hasi_pdf).round(4))
# mse_15_hasi_v="".join([mse,mse_asi_v])

# asi_v25_hasi_pdf=normal.pdf(par_est_25_hasi['v'])
# sim_asi_v25_hasi_pdf=normal.pdf(par_sim_25_hasi['v'])
# mse_asi_v = str(jsd(asi_v25_hasi_pdf, sim_asi_v25_hasi_pdf).round(4))
# mse_25_hasi_v="".join([mse,mse_asi_v])

# asi_a15_hasi_pdf=normal.pdf(par_est_15_hasi['a'])
# sim_asi_a15_hasi_pdf=normal.pdf(par_sim_15_hasi['a'])
# mse_asi_a = str(jsd(asi_a15_hasi_pdf, sim_asi_a15_hasi_pdf).round(4))
# mse_15_hasi_a="".join([mse,mse_asi_a])

# asi_a25_hasi_pdf=normal.pdf(par_est_25_hasi['a'])
# sim_asi_a25_hasi_pdf=normal.pdf(par_sim_25_hasi['a'])
# mse_asi_a = str(jsd(asi_a25_hasi_pdf, sim_asi_a25_hasi_pdf).round(4))
# mse_25_hasi_a="".join([mse,mse_asi_a])





mse_15_lasi_v = str(mean_squared_error(par_est_15_lasi['v'], 
                                        par_sim_15_lasi['v']).round(3))
mse_15_lasi_v="".join([mse,mse_15_lasi_v])
mse_15_hasi_v = str(mean_squared_error(par_est_15_hasi['v'], 
                                        par_sim_15_hasi['v']).round(3))
mse_15_hasi_v="".join([mse,mse_15_hasi_v])
mse_25_lasi_v = str(mean_squared_error(par_est_25_lasi['v'], 
                                        par_sim_25_lasi['v']).round(3))
mse_25_lasi_v="".join([mse,mse_25_lasi_v])
mse_25_hasi_v = str(mean_squared_error(par_est_25_hasi['v'], 
                                        par_sim_25_hasi['v']).round(3))
mse_25_hasi_v="".join([mse,mse_25_hasi_v])

mse_15_lasi_a = str(mean_squared_error(par_est_15_lasi['a'], 
                                        par_sim_15_lasi['a']).round(3))
mse_15_lasi_a="".join([mse,mse_15_lasi_a])
mse_15_hasi_a = str(mean_squared_error(par_est_15_hasi['a'], 
                                        par_sim_15_hasi['a']).round(3))
mse_15_hasi_a="".join([mse,mse_15_hasi_a])
mse_25_lasi_a = str(mean_squared_error(par_est_25_lasi['a'], 
                                        par_sim_25_lasi['a']).round(3))
mse_25_lasi_a="".join([mse,mse_25_lasi_a])
mse_25_hasi_a = str(mean_squared_error(par_est_25_hasi['a'], 
                                        par_sim_25_hasi['a']).round(3))
mse_25_hasi_a="".join([mse,mse_25_hasi_a])


#Cohen's d
asi_v15_lasi_pdf= (par_est_15_lasi['v'])
sim_asi_v15_lasi_pdf= (par_sim_15_lasi['v'])
csd_asi_v = str(  cohens_d(asi_v15_lasi_pdf, sim_asi_v15_lasi_pdf).round(2))
csd_15_lasi_v="".join([csd,csd_asi_v])

asi_v25_lasi_pdf= (par_est_25_lasi['v'])
sim_asi_v25_lasi_pdf= (par_sim_25_lasi['v'])
csd_asi_v = str(  cohens_d(asi_v25_lasi_pdf, sim_asi_v25_lasi_pdf).round(2))
csd_25_lasi_v="".join([csd,csd_asi_v])

asi_a15_lasi_pdf= (par_est_15_lasi['a'])
sim_asi_a15_lasi_pdf= (par_sim_15_lasi['a'])
csd_asi_a = str(  cohens_d(asi_a15_lasi_pdf, sim_asi_a15_lasi_pdf).round(2))
csd_15_lasi_a="".join([csd,csd_asi_a])

asi_a25_lasi_pdf= (par_est_25_lasi['a'])
sim_asi_a25_lasi_pdf= (par_sim_25_lasi['a'])
csd_asi_a = str(  cohens_d(asi_a25_lasi_pdf, sim_asi_a25_lasi_pdf).round(2))
csd_25_lasi_a="".join([csd,csd_asi_a])

asi_v15_hasi_pdf= (par_est_15_hasi['v'])
sim_asi_v15_hasi_pdf= (par_sim_15_hasi['v'])
csd_asi_v = str(cohens_d(asi_v15_hasi_pdf, sim_asi_v15_hasi_pdf).round(2) )
csd_15_hasi_v="".join([csd,csd_asi_v])

asi_v25_hasi_pdf= (par_est_25_hasi['v'])
sim_asi_v25_hasi_pdf= (par_sim_25_hasi['v'])
csd_asi_v = str(cohens_d(asi_v25_hasi_pdf, sim_asi_v25_hasi_pdf).round(2) )
csd_25_hasi_v="".join([csd,csd_asi_v])

asi_a15_hasi_pdf= (par_est_15_hasi['a'])
sim_asi_a15_hasi_pdf= (par_sim_15_hasi['a'])
csd_asi_a = str(cohens_d(asi_a15_hasi_pdf, sim_asi_a15_hasi_pdf).round(2) )
csd_15_hasi_a="".join([csd,csd_asi_a])

asi_a25_hasi_pdf= (par_est_25_hasi['a'])
sim_asi_a25_hasi_pdf= (par_sim_25_hasi['a'])
csd_asi_a = str(cohens_d(asi_a25_hasi_pdf, sim_asi_a25_hasi_pdf).round(2) )
csd_25_hasi_a="".join([csd,csd_asi_a])

#plot parameters
fig = make_subplots(
    rows=2, cols=2, column_titles=['Low Precision', 'High Precision',],vertical_spacing=0.1
)

colours={"estimated": "gray",
         "simulated": "goldenrod"}



# Drift Rate
#Low Precision
fig.add_trace(go.Violin(x=par_est_15_lasi['Group'][ par_est_15_lasi['Source'] == 'estimated'],
                        y=par_est_15_lasi['v'][ par_est_15_lasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_sim_15_lasi['Group'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        y=par_sim_15_lasi['v'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',
                        line_color=colours['simulated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_est_15_hasi['Group'][ par_est_15_hasi['Source'] == 'estimated'],
                        y=par_est_15_hasi['v'][ par_est_15_hasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=1)

fig.add_trace(go.Violin(x=par_sim_15_hasi['Group'][ par_sim_15_hasi['Source'] == 'simulated' ],
                        y=par_sim_15_hasi['v'][ par_sim_15_hasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=1)


fig.add_annotation(
       y=1.05
     , x='Low ASI'
     , text=mse_15_lasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)
fig.add_annotation(
       y=1
     , x='High ASI'
     , text=mse_15_hasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)


fig.add_annotation(
       y=1.12
     , x='Low ASI'
     , text=csd_15_lasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)
fig.add_annotation(
       y=1.07
     , x='High ASI'
     , text=csd_15_hasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=1)

#High Precision

fig.add_trace(go.Violin(x=par_est_25_lasi['Group'][ par_est_25_lasi['Source'] == 'estimated'],
                        y=par_est_25_lasi['v'][ par_est_25_lasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_sim_25_lasi['Group'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        y=par_sim_25_lasi['v'][ par_sim_25_lasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_est_25_hasi['Group'][ par_est_25_hasi['Source'] == 'estimated'],
                        y=par_est_25_hasi['v'][ par_est_25_hasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", showlegend=False,
                        line_color=colours['estimated']), col=2, row=1)

fig.add_trace(go.Violin(x=par_sim_25_hasi['Group'][ par_sim_25_hasi['Source'] == 'simulated' ],
                        y=par_sim_25_hasi['v'][ par_sim_25_hasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=1)

fig.add_annotation(
       y=1.5
     , x='Low ASI'
     , text=mse_25_lasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)
fig.add_annotation(
       y=1.3
     , x='High ASI'
     , text=mse_25_hasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

fig.add_annotation(
       y=1.57
     , x='Low ASI'
     , text=csd_25_lasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)
fig.add_annotation(
       y=1.37
     , x='High ASI'
     , text=csd_25_hasi_v
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=1)

#Decision theshold
#Low Precision
fig.add_trace(go.Violin(x=par_est_15_lasi['Group'][ par_est_15_lasi['Source'] == 'estimated'],
                        y=par_est_15_lasi['a'][ par_est_15_lasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_sim_15_lasi['Group'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        y=par_sim_15_lasi['a'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_est_15_hasi['Group'][ par_est_15_hasi['Source'] == 'estimated'],
                        y=par_est_15_hasi['a'][ par_est_15_hasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=1, row=2)

fig.add_trace(go.Violin(x=par_sim_15_hasi['Group'][ par_sim_15_hasi['Source'] == 'simulated' ],
                        y=par_sim_15_hasi['a'][ par_sim_15_hasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=1, row=2)


fig.add_annotation(
       y=2.5
     , x='Low ASI'
     , text=mse_15_lasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)
fig.add_annotation(
       y=2.5
     , x='High ASI'
     , text=mse_15_hasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)


fig.add_annotation(
       y=2.55
     , x='Low ASI'
     , text=csd_15_lasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)
fig.add_annotation(
       y=2.55
     , x='High ASI'
     , text=csd_15_hasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=1, row=2)

#High Precision

fig.add_trace(go.Violin(x=par_est_25_lasi['Group'][ par_est_25_lasi['Source'] == 'estimated'],
                        y=par_est_25_lasi['a'][ par_est_25_lasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',showlegend=False,
                                                legendgrouptitle_text="Source", 
                        line_color=colours['estimated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_sim_25_lasi['Group'][ par_sim_15_lasi['Source'] == 'simulated' ],
                        y=par_sim_25_lasi['a'][ par_sim_25_lasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_est_25_hasi['Group'][ par_est_25_hasi['Source'] == 'estimated'],
                        y=par_est_25_hasi['a'][ par_est_25_hasi['Source'] == 'estimated' ],
                        legendgroup='source', scalegroup='source', name='estimated',
                                                legendgrouptitle_text="Source", showlegend=False,
                        line_color=colours['estimated']), col=2, row=2)

fig.add_trace(go.Violin(x=par_sim_25_hasi['Group'][ par_sim_25_hasi['Source'] == 'simulated' ],
                        y=par_sim_25_hasi['a'][ par_sim_25_hasi['Source'] == 'simulated' ],
                        legendgroup='source', scalegroup='source', name='simulated',showlegend=False,
                        line_color=colours['simulated']), col=2, row=2)

fig.add_annotation(
       y=2.88
     , x='Low ASI'
     , text=mse_25_lasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)
fig.add_annotation(
       y=2.8
     , x='High ASI'
     , text=mse_25_hasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)

fig.add_annotation(
       y=2.95
     , x='Low ASI'
     , text=csd_25_lasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)
fig.add_annotation(
       y=2.87
     , x='High ASI'
     , text=csd_25_hasi_a
     ,showarrow=False
     , font=dict(size=18, color="black", family="Courier New"),
    col=2, row=2)




fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')
fig.update_annotations(font_size=50)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

opac=0.6


#add line to show interaction
#ASI

fig.add_shape(type='line',
                x0='Low ASI',
                y0=mva_coh_asi_traces['v(low_asi.15)'].mean(),
                x1='High ASI',
                y1=mva_coh_asi_traces['v(high_asi.15)'].mean(),
                line=dict(color=colours['estimated'], dash='dash'),opacity=opac,
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=va_asi_coh_sim_traces['v(low_asi.15)'].mean(),
                x1='High ASI',
                y1=va_asi_coh_sim_traces['v(high_asi.15)'].mean(),
                line=dict(color=colours['simulated']),opacity=opac,
                row=1,
                col=1)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=mva_coh_asi_traces['v(low_asi.25)'].mean(),
                x1='High ASI',
                y1=mva_coh_asi_traces['v(high_asi.25)'].mean(),
                line=dict(color=colours['estimated']),opacity=opac,
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=va_asi_coh_sim_traces['v(low_asi.25)'].mean(),
                x1='High ASI',
                y1=va_asi_coh_sim_traces['v(high_asi.25)'].mean(),
                line=dict(color=colours['simulated'], dash='dash'),opacity=opac,
                row=1,
                col=2)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=mva_coh_asi_traces['a(low_asi.15)'].mean(),
                x1='High ASI',
                y1=mva_coh_asi_traces['a(high_asi.15)'].mean(),
                line=dict(color=colours['estimated'], dash='dash'),opacity=opac,
                row=2,
                col=1)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=va_asi_coh_sim_traces['a(low_asi.15)'].mean(),
                x1='High ASI',
                y1=va_asi_coh_sim_traces['a(high_asi.15)'].mean(),
                line=dict(color=colours['simulated'], dash='dash'),opacity=opac,
                row=2,
                col=1)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=mva_coh_asi_traces['a(low_asi.25)'].mean(),
                x1='High ASI',
                y1=mva_coh_asi_traces['a(high_asi.25)'].mean(),
                line=dict(color=colours['estimated'], dash='dash'),opacity=opac,
                row=2,
                col=2)

fig.add_shape(type='line',
                x0='Low ASI',
                y0=va_asi_coh_sim_traces['a(low_asi.25)'].mean(),
                x1='High ASI',
                y1=va_asi_coh_sim_traces['a(high_asi.25)'].mean(),
                line=dict(color=colours['simulated'], dash='dash'),opacity=opac,
                row=2,
                col=2)

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


# Update yaxis properties
fig.update_yaxes(title_text="Drift rate", col=1, row=1)
fig.update_yaxes(title_text="Drift rate", col=2, row=1)
fig.update_yaxes(title_text="Decision threshold", col=1, row=2)
fig.update_yaxes(title_text="Decision threshold", col=2, row=2)


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font,
    title_font_color="blue", 
    legend_title_font_color="blue",
    legend_font_size=size_font,
    autosize=False,
    width=2800,
    height=2000,
)
fig.update_xaxes(title_font_family="Arial")
fig.write_image("study1_fig_asi_group_recovery_2024.png",scale=3)

fig.show()

