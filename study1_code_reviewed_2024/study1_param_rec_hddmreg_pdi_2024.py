#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:00:54 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:37:53 2022

@author: francescoscaramozzino
"""
import hddm
import matplotlib.pyplot as plt 
import pandas as pd

import seaborn as sns

import numpy as np


#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

#all 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_quantiles
#%%

data= hddm.load_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

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
bstd_v_pdi=stats_s.at['std','v_z_PDI']*data.PDI.std() #destandardising beta

#condition
b_v_cond=stats_s.at['mean','v_coherence[T.25]']
bstd_v_cond=stats_s.at['std','v_coherence[T.25]']

###decision threshold
int_a=stats_s.at['mean','a_Intercept']
int_a_sd=stats_s.at['std','a_Intercept']

#PDI
b_a_pdi=stats_s.at['mean','a_z_PDI']
bstd_a_pdi=stats_s.at['std','a_z_PDI']*data.PDI.std() #destandardising beta

#condition

b_a_cond=stats_s.at['mean','a_coherence[T.25]']
bstd_a_cond=stats_s.at['std','a_coherence[T.25]']

###t
t=stats_s.at['mean','t']
t_std=stats_s.at['std','t']

#%%
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
    
    a_std = int_a_sd+bstd_a_pdi*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v_std = int_v_sd+bstd_v_pdi*xx  # can also do for drift, here using same beta coeff
    
    a25_std = int_a_sd +bstd_a_cond+bstd_a_pdi*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v25_std = int_v_sd+bstd_v_cond+bstd_v_pdi*xx  # can also do for drift, here using same beta coeff
    
  
parvec_cond_0 = {'v':v.mean()  , 'a':a.mean()  , 't':t,
                 'sv':v_std.mean(), 
                 'sa':a_std.mean(), 
                 "st":t_std.mean()}
parvec_cond_1 = {'v':v25.mean()  , 'a':a25.mean()  , 't':t,
                 'sv':v25_std.mean(), 
                 'sa':a25_std.mean(), 
                 "st":t_std.mean()} # set a to value set by regression, here v is set to constant

# note that for subjs_per_bin > 1, these are just the mean values of the parameters; indiv subjs within bin are sampled from distributions with the given means, but can still differ within bin around those means. 
#not including sv, sz, st in the statement ensures those are actually 0.

data_sim, params_sim = hddm.generate.gen_rand_data({'level_cond_0': parvec_cond_0,
                                                        'level_cond_1': parvec_cond_1}, 
                                                       size=trials_per_level, 
                                                       subjs=subjs_per_bin)
    
    # can also do with two levels of within-subj conditions
    # data_a, params_a = hddm.generate.gen_rand_data({'level1': parvec,'level2': parvec2}, size=trials_per_level, subjs=subjs_per_bin)

#simulating PDI  values by randomly sampling from our data

pdi_sam=data.PDI.dropna()
f=pd.Series(pdi_sam.sample(n=191,replace=True)).reset_index()
pdi_sim=pd.concat([f]*30, ignore_index=True)#to have a value of CAPS for each trial
pdi_sim=pdi_sim['PDI']
data_sim['z_pdi']=(pdi_sim - pdi_sim.mean())/pdi_sim.std(ddof=0)

data_sim=data_sim[['rt','response','condition','z_pdi']]

data_sim.loc[data_sim['condition'] == 'level_cond_0', 'coherence'] = '15'
data_sim.loc[data_sim['condition'] == 'level_cond_0', 'coherence'] = '15'
data_sim.loc[data_sim['condition'] == 'level_cond_1', 'coherence'] = '25'
data_sim.loc[data_sim['condition'] == 'level_cond_1', 'coherence'] = '25'



data_sim=data_sim[['rt','response','coherence','z_pdi']].dropna()
data_sim= data_sim[(data_sim['rt'] > 0.3)].assign(rt_type = "simulated")
data_sim = data_sim[(data_sim['rt'] < 6)] 


#%%


#cond,PDI
sim_mreg_av_cond_PDI2= hddm.models.HDDMRegressor(data_sim, ['a ~ 1+coherence+z_pdi', 
                                                            'v ~ 1+coherence+z_pdi'],
                                                 group_only_regressors=True, p_outlier=0.10)
sim_mreg_av_cond_PDI2.find_starting_values()
sim_mreg_av_cond_PDI2.sample(20000, burn=2000,thin=5, dbname='sim_hddm_lr_c_pdi_traces.db', db='pickle')
sim_mreg_av_cond_PDI2.save('sim_hddm_lr_c_pdi')
#%%
sim_mreg_av_cond_PDI = hddm.load('sim_hddm_lr_c_pdi')

sim_mreg_av_cond_PDI.print_stats()




sim_a_PDI= sim_mreg_av_cond_PDI.nodes_db.node ['a_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([sim_a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_PDI.trace() > 0).mean())

sim_v_PDI = sim_mreg_av_cond_PDI.nodes_db.node ['v_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([sim_v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_PDI.trace() < 0).mean())

sim_a_cond= sim_mreg_av_cond_PDI.nodes_db.node ['a_coherence[T.25]'] 
hddm.analyze.plot_posterior_nodes ([sim_a_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_cond.trace() > 0).mean())

sim_v_cond = sim_mreg_av_cond_PDI.nodes_db.node ['v_coherence[T.25]'] 
hddm.analyze.plot_posterior_nodes ([sim_v_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_cond.trace() < 0).mean())



a_PDI= mreg_av_cond_PDI.nodes_db.node ['a_z_PDI'] 
v_PDI = mreg_av_cond_PDI.nodes_db.node ['v_z_PDI'] 
a_cond= mreg_av_cond_PDI.nodes_db.node ['a_coherence[T.25]'] 
v_cond = mreg_av_cond_PDI.nodes_db.node ['v_coherence[T.25]'] 
#%%
#plot betas
mreg_av_cond_PDI_traces=mreg_av_cond_PDI.get_traces()

a_z_pdi=mreg_av_cond_PDI_traces['a_z_PDI']
v_z_pdi=mreg_av_cond_PDI_traces['v_z_PDI']

a_z_cond=mreg_av_cond_PDI_traces['a_coherence[T.25]']
v_z_cond=mreg_av_cond_PDI_traces['v_coherence[T.25]']


sim_mreg_av_cond_PDI_traces=sim_mreg_av_cond_PDI.get_traces()


sim_a_pdi=sim_mreg_av_cond_PDI_traces['a_z_pdi']
sim_v_pdi=sim_mreg_av_cond_PDI_traces['v_z_pdi']

sim_a_cond=sim_mreg_av_cond_PDI_traces['a_coherence[T.25]']
sim_v_cond=sim_mreg_av_cond_PDI_traces['v_coherence[T.25]']
#%%
#plot betas for PDI
dic_pdi_v={'Est. β for v':v_z_pdi, 'Sim. β for v':sim_v_pdi}
df_pdi_v = pd.DataFrame(dic_pdi_v)

dic_pdi_a={'Est. β for a':a_z_pdi,'Sim. β for a':sim_a_pdi}
df_pdi_a = pd.DataFrame(dic_pdi_a)


sns.set_style('whitegrid')

ax=sns.displot(df_pdi_v, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value PDI - P(est. v < sim. v) =  0.06',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_pdi_v, fontsize=28 )

print ("P(est. v < sim. v) = ", (v_PDI.trace()  < sim_v_PDI.trace()).mean())


ax=sns.displot(df_pdi_a, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value PDI - P(est. a < sim. a) =  0.72',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_pdi_a, fontsize=28 )

print ("P(v_PDI < 0) = ", (a_PDI.trace()  < sim_a_PDI.trace()).mean())

#%%

#plot betas for Cond
dic_cond_v={'Est. β for v':v_z_cond, 'Sim. β for v':sim_v_cond}
df_cond_v = pd.DataFrame(dic_cond_v)

dic_cond_a={'Est. β for a':a_z_cond,'Sim. β for a':sim_a_cond}
df_cond_a = pd.DataFrame(dic_cond_a)


sns.set_style('whitegrid')

ax=sns.displot(df_cond_v, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value condition - P(est. v < sim. v) =  0.18',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_cond_v, fontsize=28 )
print ("P(est. v < sim. v) = ", (v_cond.trace()  < sim_v_cond.trace()).mean())

ax=sns.displot(df_cond_a, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value condition - P(est. a < sim. a) =  0.45',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_cond_a, fontsize=28 )
print ("P(est. v < sim. v) = ", (a_cond.trace()  < sim_a_cond.trace()).mean())

#%%
#CAPS

mreg_av_cond_caps = hddm.load('hddm_lr_c_caps')

mreg_av_cond_caps.print_stats()
statsy=mreg_av_cond_caps.gen_stats()
stats_s=statsy.T

mreg_av_cond_caps_traces=mreg_av_cond_caps.get_traces()



###drift rate
int_v=stats_s.at['mean','v_Intercept']
int_v_sd=stats_s.at['std','v_Intercept']
 
#caps
b_v_caps=stats_s.at['mean','v_z_CAPS']*data.caps.std() #destandardising beta
bstd_v_caps=stats_s.at['std','v_z_CAPS']*data.caps.std() #destandardising beta

#condition
b_v_cond=stats_s.at['mean','v_coherence[T.25]']
bstd_v_cond=stats_s.at['std','v_coherence[T.25]']

###decision threshold
int_a=stats_s.at['mean','a_Intercept']
int_a_sd=stats_s.at['std','a_Intercept']

#caps
b_a_caps=stats_s.at['mean','a_z_CAPS']
bstd_a_caps=stats_s.at['std','a_z_CAPS']*data.caps.std() #destandardising beta

#condition

b_a_cond=stats_s.at['mean','a_coherence[T.25]']
bstd_a_cond=stats_s.at['std','a_coherence[T.25]']

###t
t=stats_s.at['mean','t']
t_std=stats_s.at['std','t']

#%%
#Generate data


trials_per_level = 15
subjs_per_bin=191


caps=data.caps



for x in caps:
    xx = (caps - caps.mean()) / caps.std()  # z-score the x factor
    
    a = int_a+b_a_caps*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v = int_v+b_v_caps*xx  # can also do for drift, here using same beta coeff
    
    a25 = int_a +b_a_cond+b_a_caps*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v25 = int_v+b_v_cond+b_v_caps*xx  # can also do for drift, here using same beta coeff
    
    a_std = int_a_sd+bstd_a_caps*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v_std = int_v_sd+bstd_v_caps*xx  # can also do for drift, here using same beta coeff
    
    a25_std = int_a_sd +bstd_a_cond+bstd_a_caps*xx  #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v25_std = int_v_sd+bstd_v_cond+bstd_v_caps*xx  # can also do for drift, here using same beta coeff
    
  
parvec_cond_0 = {'v':v.mean()  , 'a':a.mean()  , 't':t,
                 'sv':v_std.mean(), 
                 'sa':a_std.mean(), 
                 "st":t_std.mean()}
parvec_cond_1 = {'v':v25.mean()  , 'a':a25.mean()  , 't':t,
                 'sv':v25_std.mean(), 
                 'sa':a25_std.mean(), 
                 "st":t_std.mean()} # set a to value set by regression, here v is set to constant

# note that for subjs_per_bin > 1, these are just the mean values of the parameters; indiv subjs within bin are sampled from distributions with the given means, but can still differ within bin around those means. 
#not including sv, sz, st in the statement ensures those are actually 0.

data_sim, params_sim = hddm.generate.gen_rand_data({'level_cond_0': parvec_cond_0,
                                                        'level_cond_1': parvec_cond_1}, 
                                                       size=trials_per_level, 
                                                       subjs=subjs_per_bin)
    
    # can also do with two levels of within-subj conditions
    # data_a, params_a = hddm.generate.gen_rand_data({'level1': parvec,'level2': parvec2}, size=trials_per_level, subjs=subjs_per_bin)

#simulating caps  values by randomly sampling from our data

caps_sam=data.caps.dropna()
f=pd.Series(caps_sam.sample(n=subjs_per_bin,replace=True)).reset_index()
caps_sim=pd.concat([f]*30, ignore_index=True)#to have a value of CAPS for each trial
caps_sim=caps_sim['caps']
data_sim['z_caps']=(caps_sim - caps_sim.mean())/caps_sim.std(ddof=0)

data_sim=data_sim[['rt','response','condition','z_caps']]

data_sim.loc[data_sim['condition'] == 'level_cond_0', 'coherence'] = '15'
data_sim.loc[data_sim['condition'] == 'level_cond_0', 'coherence'] = '15'
data_sim.loc[data_sim['condition'] == 'level_cond_1', 'coherence'] = '25'
data_sim.loc[data_sim['condition'] == 'level_cond_1', 'coherence'] = '25'



data_sim=data_sim[['rt','response','coherence','z_caps']].dropna()
data_sim= data_sim[(data_sim['rt'] > 0.3)].assign(rt_type = "simulated")
data_sim = data_sim[(data_sim['rt'] < 6)] 


#%%


#cond,caps
sim_mreg_av_cond_caps2= hddm.models.HDDMRegressor(data_sim, ['a ~ 1+coherence+z_caps', 
                                                            'v ~ 1+coherence+z_caps'],
                                                 group_only_regressors=True, p_outlier=0.10)
sim_mreg_av_cond_caps2.find_starting_values()
sim_mreg_av_cond_caps2.sample(20000, burn=2000,thin=5, dbname='sim_hddm_lr_c_caps_traces.db', db='pickle')
sim_mreg_av_cond_caps2.save('sim_hddm_lr_c_caps')
sim_mreg_av_cond_caps2.print_stats()

#%%
sim_mreg_av_cond_PDI = hddm.load('sim_hddm_lr_c_pdi')

sim_mreg_av_cond_PDI.print_stats()




sim_a_PDI= sim_mreg_av_cond_PDI.nodes_db.node ['a_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([sim_a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_PDI.trace() > 0).mean())

sim_v_PDI = sim_mreg_av_cond_PDI.nodes_db.node ['v_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([sim_v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_PDI.trace() < 0).mean())

sim_a_cond= sim_mreg_av_cond_PDI.nodes_db.node ['a_coherence[T.25]'] 
hddm.analyze.plot_posterior_nodes ([sim_a_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_cond.trace() > 0).mean())

sim_v_cond = sim_mreg_av_cond_PDI.nodes_db.node ['v_coherence[T.25]'] 
hddm.analyze.plot_posterior_nodes ([sim_v_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_cond.trace() < 0).mean())



a_PDI= mreg_av_cond_PDI.nodes_db.node ['a_z_PDI'] 
v_PDI = mreg_av_cond_PDI.nodes_db.node ['v_z_PDI'] 
a_cond= mreg_av_cond_PDI.nodes_db.node ['a_coherence[T.25]'] 
v_cond = mreg_av_cond_PDI.nodes_db.node ['v_coherence[T.25]'] 
#%%
#plot betas
mreg_av_cond_PDI_traces=mreg_av_cond_PDI.get_traces()

a_z_pdi=mreg_av_cond_PDI_traces['a_z_PDI']
v_z_pdi=mreg_av_cond_PDI_traces['v_z_PDI']

a_z_cond=mreg_av_cond_PDI_traces['a_coherence[T.25]']
v_z_cond=mreg_av_cond_PDI_traces['v_coherence[T.25]']


sim_mreg_av_cond_PDI_traces=sim_mreg_av_cond_PDI.get_traces()


sim_a_pdi=sim_mreg_av_cond_PDI_traces['a_z_pdi']
sim_v_pdi=sim_mreg_av_cond_PDI_traces['v_z_pdi']

sim_a_cond=sim_mreg_av_cond_PDI_traces['a_coherence[T.25]']
sim_v_cond=sim_mreg_av_cond_PDI_traces['v_coherence[T.25]']
#%%
#plot betas for PDI
dic_pdi_v={'Est. β for v':v_z_pdi, 'Sim. β for v':sim_v_pdi}
df_pdi_v = pd.DataFrame(dic_pdi_v)

dic_pdi_a={'Est. β for a':a_z_pdi,'Sim. β for a':sim_a_pdi}
df_pdi_a = pd.DataFrame(dic_pdi_a)


sns.set_style('whitegrid')

ax=sns.displot(df_pdi_v, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value PDI - P(est. v < sim. v) =  0.06',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_pdi_v, fontsize=28 )

print ("P(est. v < sim. v) = ", (v_PDI.trace()  < sim_v_PDI.trace()).mean())


ax=sns.displot(df_pdi_a, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value PDI - P(est. a < sim. a) =  0.72',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_pdi_a, fontsize=28 )

print ("P(v_PDI < 0) = ", (a_PDI.trace()  < sim_a_PDI.trace()).mean())

#%%

#plot betas for Cond
dic_cond_v={'Est. β for v':v_z_cond, 'Sim. β for v':sim_v_cond}
df_cond_v = pd.DataFrame(dic_cond_v)

dic_cond_a={'Est. β for a':a_z_cond,'Sim. β for a':sim_a_cond}
df_cond_a = pd.DataFrame(dic_cond_a)


sns.set_style('whitegrid')

ax=sns.displot(df_cond_v, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value condition - P(est. v < sim. v) =  0.18',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_cond_v, fontsize=28 )
print ("P(est. v < sim. v) = ", (v_cond.trace()  < sim_v_cond.trace()).mean())

ax=sns.displot(df_cond_a, kind="kde", fill=True, bw_adjust=2,height=8, aspect=2.2)
plt.xlabel('β value condition - P(est. a < sim. a) =  0.45',size=28, weight="bold",labelpad=24)
plt.ylabel('Posterior Probability Density',size=28, weight="bold", labelpad=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(df_cond_a, fontsize=28 )
print ("P(est. v < sim. v) = ", (a_cond.trace()  < sim_a_cond.trace()).mean())
