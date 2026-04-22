#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:22:35 2024

@author: francescoscaramozzino
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:24:18 2022

@author: PHJT002
"""
import hddm
import matplotlib.pyplot as plt 
import pandas as pd


#%%

#load data and clean outliers as in prereg

data3=pd.read_csv('data_study1.csv')
data3 = data3[(data3['rt'] > 0.3)] 
data3 = data3[(data3['rt'] < 6)] 

#z-scoring the variable

data3['coherence']=data3['coherence'].astype('object')

data3['z_PDI']=(data3.PDI - data3.PDI.mean())/data3.PDI.std()

data3['z_CAPS']=(data3.caps - data3.caps.mean())/data3.caps.std(ddof=0)

data3['z_ASI']=(data3.asi - data3.asi.mean())/data3.asi.std(ddof=0)

#%%
#models compared to Null-model perfomed in study1_betweengroupModles_2024.py


#  v and a <-- Motion coherence

hddm_lr_c= hddm.models.HDDMRegressor(data3, ['a ~ 1+coherence', 
                                                    'v ~ 1+coherence'], 
                                            group_only_regressors=True, p_outlier=0.10)
hddm_lr_c.find_starting_values()
hddm_lr_c.sample(20000, burn=2000,thin=5, dbname='hddm_lr_c_traces.db', db='pickle')
hddm_lr_c.save('hddm_lr_c')

hddm_lr_c = hddm.load('hddm_lr_c')

hddm_lr_c.print_stats()

# DIC: 16080.465042
# deviance: 16075.457611
# pD: 5.007431

#%%
#  v and a <-- Motion coherence + PDI 
hddm_lr_c_pdi= hddm.models.HDDMRegressor(data3, ['a ~ 1+coherence+z_PDI', 
                                                    'v ~ 1+coherence+z_PDI'], 
                                            group_only_regressors=True, p_outlier=0.10)
hddm_lr_c_pdi.find_starting_values()
hddm_lr_c_pdi.sample(20000, burn=2000,thin=5, dbname='hddm_lr_c_pdi_traces.db', db='pickle')
hddm_lr_c_pdi.save('hddm_lr_c_pdi')

#%%
hddm_lr_c_pdi = hddm.load('hddm_lr_c_pdi')

hddm_lr_c_pdi.print_stats()

# hddm_lr_c_pdi = hddm.load('mreg_av_PDI')

# DIC: 16062.889233
# deviance: 16055.964261
# pD: 6.924972

#%%
#  v and a <-- Motion coherence + CAPS 

#################
hddm_lr_c_caps= hddm.models.HDDMRegressor(data3,  ['a ~ 1+coherence+z_CAPS', 
                                                    'v ~ 1+coherence+z_CAPS'], group_only_regressors=True, p_outlier=0.10)
hddm_lr_c_caps.find_starting_values()
hddm_lr_c_caps.sample(20000, burn=2000,thin=5, dbname='hddm_lr_c_caps_traces.db', db='pickle')
hddm_lr_c_caps.save('hddm_lr_c_caps')

#%%
hddm_lr_c_caps = hddm.load('hddm_lr_c_caps')

hddm_lr_c_caps.print_stats()
# DIC: 16050.718666
# deviance: 16043.673697
# pD: 7.044969


#%%

#  v and a <-- Motion coherence + ASI 


hddm_lr_c_asi= hddm.models.HDDMRegressor(data3, ['a ~ 1+coherence+z_ASI', 
                                                    'v ~ 1+coherence+z_ASI'], group_only_regressors=True,  p_outlier=0.10)
hddm_lr_c_asi.find_starting_values()
hddm_lr_c_asi.sample(20000, burn=2000,thin=5, dbname='hddm_lr_c_asi_traces.db', db='pickle')
hddm_lr_c_asi.save('hddm_lr_c_asi')

#%%
hddm_lr_c_asi = hddm.load('hddm_lr_c_asi')

hddm_lr_c_asi.print_stats()

# DIC: 16083.336794
# deviance: 16076.240763
# pD: 7.096030


#%%

#Resutls PDI

a_PDI= hddm_lr_c_pdi.nodes_db.node ['a_z_PDI'] 
hddm.analyze.plot_posterior_nodes ([a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI.trace() > 0).mean())

v_PDI = hddm_lr_c_pdi.nodes_db.node ['v_z_PDI'] 
hddm.analyze.plot_posterior_nodes ([v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI.trace() < 0).mean())


#%%

#Resutls CAPS

a_CAPS= hddm_lr_c_caps.nodes_db.node ['a_z_CAPS'] 
hddm.analyze.plot_posterior_nodes ([a_CAPS], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() < 0).mean())

v_CAPS = hddm_lr_c_caps.nodes_db.node ['v_z_CAPS'] 
hddm.analyze.plot_posterior_nodes ([v_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() > 0).mean())

#%%

#Resutls ASI

#ASI

a_ASI= hddm_lr_c_asi.nodes_db.node ['a_z_ASI'] 
hddm.analyze.plot_posterior_nodes ([a_ASI], bins=8)
plt.legend(['β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_ASI > 0) = ", (a_ASI.trace() > 0).mean())

v_ASI = hddm_lr_c_asi.nodes_db.node ['v_z_ASI'] 
hddm.analyze.plot_posterior_nodes ([v_ASI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_ASI > 0) = ", (v_ASI.trace() > 0).mean())
print ("P(v_ASI > 0) = ", (v_ASI.trace() < 0).mean())

