#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:53:47 2023

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import py_stringmatching as sm

#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

data= hddm.load_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 


#%%
hddm_pdi = hddm.HDDM(data,depends_on={'v': ['pdi_group'], 
                                                'a':['pdi_group']}, 
                       std_depends=True, p_outlier=0.10)
hddm_pdi.find_starting_values()
hddm_pdi.sample(20000, burn=2000, thin=5,  dbname='hddm_pdi_traces.db', db='pickle')
hddm_pdi.save('hddm_pdi')
#%%
hddm_pdi= hddm.load('hddm_pdi') 

hddm_pdi.print_stats()

# DIC: 16077.218549
# deviance: 16068.219715
# pD: 8.998833

#%%
hddm_caps = hddm.HDDM(data,depends_on={'v': ['caps_group'], 
                                                'a':['caps_group']}, std_depends=True, p_outlier=0.10)
hddm_caps.find_starting_values()
hddm_caps.sample(20000, burn=2000, thin=5,  dbname='hddm_caps_traces.db', db='pickle')
hddm_caps.save('hddm_caps')
#%%


hddm_caps= hddm.load('hddm_caps') 
hddm_caps.print_stats()

#%%
###########
#HDDM results
v_L, v_H = hddm_caps.nodes_db.node[['v(low_caps)', 'v(high_caps)']]
caps_vposteriors= hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=10)
plt.legend(['v Lcaps','v Hcaps'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(H >L)=",(v_H.trace()> v_L.trace()).mean())
print ("P(L >H)=", (v_L.trace()> v_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['v_H'], ['v_L'])

a_L, a_H = hddm_caps.nodes_db.node[['a(low_caps)', 'a(high_caps)']]
caps_aposteriors= hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=10)
plt.legend(['a Lcaps','a Hcaps'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(H <L)=",(a_H.trace()< a_L.trace()).mean())
print ("P(L <H)=", (a_L.trace()< a_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['a_L'], ['v_L'])





# DIC: 16067.085984
# deviance: 16058.178657
# pD: 8.907327
#%%
hddm_asi = hddm.HDDM(data,depends_on={'v': ['asi_group'], 
                                                'a':['asi_group']}, std_depends=True, p_outlier=0.10)
hddm_asi.find_starting_values()
hddm_asi.sample(20000, burn=2000, thin=5,  dbname='hddm_asi_traces.db', db='pickle')
hddm_asi.save('hddm_c_asi')

hddm_asi= hddm.load('hddm_c_asi') 
hddm_asi.print_stats()

# DIC: 16078.927262
# deviance: 16069.967960
# pD: 8.959302

#%%
#  v and a X PDI
va_null_15 = hddm.HDDM(data[data.coherence == 15], std_depends=True, p_outlier=0.10)
va_null_15.find_starting_values()
va_null_15.sample(20000, burn=2000, thin=5, dbname='va_null_15_traces.db', db='pickle')
va_null_15.save('va_null_15')

va_null_15.print_stats()

#%%
#  v and a X PDI
va_null_25 = hddm.HDDM(data[data.coherence == 25], std_depends=True, p_outlier=0.10)
va_null_25.find_starting_values()
va_null_25.sample(20000, burn=2000, thin=5, dbname='va_null_25_traces.db', db='pickle')
va_null_25.save('va_null_25')

va_null_25.print_stats()

#%%
#  v and a X PDI
va_PDI_15 = hddm.HDDM(data[data.coherence == 15],depends_on={'v': 'pdi_group', 'a':'pdi_group'}, std_depends=True, p_outlier=0.10)
va_PDI_15.find_starting_values()
va_PDI_15.sample(20000, burn=2000, thin=5, dbname='va_PDI_15_traces.db', db='pickle')
va_PDI_15.save('va_PDI_15')


va_PDI_15.print_stats()
va_PDI_15=hddm.load('va_PDI_15')

va_PDI_15.plot_posteriors()

va_PDI_15.plot_posterior_predictive()


#%%
va_PDI_15=hddm.load('va_PDI_15')

###########
#HDDM results
v_L, v_H = va_PDI_15.nodes_db.node[['v(low_pdi)', 'v(high_pdi)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=10)
plt.legend(['v LPDI','v HPDI'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(H >L)=",(v_H.trace()> v_L.trace()).mean())
print ("P(L >H)=", (v_L.trace()> v_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['v_H'], ['v_L'])

a_L, a_H = va_PDI_15.nodes_db.node[['a(low_pdi)', 'a(high_pdi)']]
PDI_aposteriors= hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=10)
plt.legend(['a LPDI','a HPDI'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(H <L)=",(a_H.trace()< a_L.trace()).mean())
print ("P(L <H)=", (a_L.trace()< a_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['a_L'], ['v_L'])
#%%
#  v and a X PDI
va_PDI_25 = hddm.HDDM(data[data.coherence == 25],depends_on={'v': 'pdi_group', 'a':'pdi_group'}, std_depends=True, p_outlier=0.10)
va_PDI_25.find_starting_values()
va_PDI_25.sample(20000, burn=2000, thin=5, dbname='va_PDI_25_traces.db', db='pickle')
va_PDI_25.save('va_PDI_25')


va_PDI_25.print_stats()
va_PDI_25=hddm.load('va_PDI_25')

va_PDI_25.plot_posteriors()

va_PDI_25.plot_posterior_predictive()


#%%

###########
va_PDI_25=hddm.load('va_PDI_25')

#HDDM results
v_L, v_H = va_PDI_25.nodes_db.node[['v(low_pdi)', 'v(high_pdi)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=10)
plt.legend(['v LPDI','v HPDI'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(H >L)=",(v_H.trace()> v_L.trace()).mean())
print ("P(L >H)=", (v_L.trace()> v_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['v_H'], ['v_L'])

a_L, a_H = va_PDI_25.nodes_db.node[['a(low_pdi)', 'a(high_pdi)']]
PDI_aposteriors= hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=10)
plt.legend(['a LPDI','a HPDI'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(H <L)=",(a_H.trace()< a_L.trace()).mean())
print ("P(L <H)=", (a_L.trace()< a_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['a_L'], ['v_L'])

#########
#Get traces of parameters for plotting

va_PDI_traces = va_PDI_25.get_traces()

v_1_pdi=va_PDI_traces[['v(low_pdi)','v(high_pdi)']]
a_1_pdi=va_PDI_traces[['a(low_pdi)','a(high_pdi)']]


#################
#Violin plot



#v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_1_pdi, palette='PiYG_r',linewidth=0.5, inner='point', split=True)
#ax.set_xlabel("PDI Groups", size=100, weight="bold",labelpad=40)
ax.set_ylabel("Drift-rate", size=100, labelpad=30)
ax.set_xticklabels(['Low PDI','High PDI'],size=100)


# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_1_pdi['v(high_pdi)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=48)
 
v_L, v_H = va_PDI_25.nodes_db.node[['v(low_pdi)', 'v(high_pdi)']]
print ("P(H <L)=",(v_H.trace()< v_L.trace()).mean())


#a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_1_pdi, palette='PiYG_r',linewidth=0.5, inner='point', split=True)
#ax.set_xlabel("PDI Groups", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Threshold", size=100, labelpad=30)
ax.set_xticklabels(['Low PDI','High PDI'],size=100)

# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_1_pdi['a(low_pdi)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col,fontsize=48)

a_L, a_H = va_PDI_25.nodes_db.node[['a(low_pdi)', 'a(HPDI)']]
print ("P(H >L)=",(a_H.trace()> a_L.trace()).mean())







#%%
#  v and a X caps
va_caps_15 = hddm.HDDM(data[data.coherence == 15],depends_on={'v': 'caps_group', 'a':'caps_group'}, std_depends=True, p_outlier=0.10)
va_caps_15.find_starting_values()
va_caps_15.sample(20000, burn=2000, thin=5, dbname='va_caps_15_traces.db', db='pickle')
va_caps_15.save('va_caps_15')


va_caps_15.print_stats()
va_caps_15=hddm.load('va_caps_15')

va_caps_15.plot_posteriors()

va_caps_15.plot_posterior_predictive()

#%%


###########
#HDDM results
v_L, v_H = va_caps_15.nodes_db.node[['v(low_caps)', 'v(high_caps)']]
caps_vposteriors= hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=10)
plt.legend(['v Lcaps','v Hcaps'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(H >L)=",(v_H.trace()> v_L.trace()).mean())
print ("P(L >H)=", (v_L.trace()> v_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['v_H'], ['v_L'])

a_L, a_H = va_caps_15.nodes_db.node[['a(low_caps)', 'a(high_caps)']]
caps_aposteriors= hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=10)
plt.legend(['a Lcaps','a Hcaps'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(H <L)=",(a_H.trace()< a_L.trace()).mean())
print ("P(L <H)=", (a_L.trace()< a_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['a_L'], ['v_L'])
#%%

#  v and a X caps
va_caps_25 = hddm.HDDM(data[data.coherence == 25],depends_on={'v': 'caps_group', 'a':'caps_group'}, std_depends=True, p_outlier=0.10)
va_caps_25.find_starting_values()
va_caps_25.sample(20000, burn=2000, thin=5, dbname='va_caps_25_traces.db', db='pickle')
va_caps_25.save('va_caps_25')


va_caps_25.print_stats()
va_caps_25=hddm.load('va_caps_25')

va_caps_25.plot_posteriors()

va_caps_25.plot_posterior_predictive()

#%%


###########
#HDDM results
v_L, v_H = va_caps_25.nodes_db.node[['v(low_caps)', 'v(high_caps)']]
caps_vposteriors= hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=10)
plt.legend(['v Lcaps','v Hcaps'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(H >L)=",(v_H.trace()> v_L.trace()).mean())
print ("P(L >H)=", (v_L.trace()> v_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['v_H'], ['v_L'])

a_L, a_H = va_caps_25.nodes_db.node[['a(low_caps)', 'a(high_caps)']]
caps_aposteriors= hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=10)
plt.legend(['a Lcaps','a Hcaps'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(H <L)=",(a_H.trace()< a_L.trace()).mean())
print ("P(L <H)=", (a_L.trace()< a_H.trace()).mean())

sm.OverlapCoefficient().get_raw_score(['a_L'], ['v_L'])
#%%

#################
#Violin plot

#v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_1_caps, palette='Spectral_r',linewidth=0.5, inner='point', split=True)
#ax.set_xlabel("CAPS Groups", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Drift-rate", size=100,labelpad=30)
ax.set_xticklabels(['Low CAPS','High CAPS'],size=100)


# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_1_caps['v(high_caps)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=48)
 
v_L, v_H = va_CAPS.nodes_db.node[['v(low_caps)', 'v(high_caps)']]
print ("P(H <L)=",(v_H.trace()< v_L.trace()).mean())


#a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_1_caps, palette='Spectral_r',linewidth=0.5, inner='point', split=True)
#ax.set_xlabel("CAPS Groups", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Threshold", size=100,labelpad=30)
ax.set_xticklabels(['Low CAPS','High CAPS'],size=100)

# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_1_caps['a(low_caps)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=48)

a_L, a_H = va_CAPS.nodes_db.node[['a(low_caps)', 'a(high_caps)']]
print ("P(H >L)=",(a_H.trace()> a_L.trace()).mean())

#%%



##############################
#v and a X ASI 


va_ASI_15 = hddm.HDDM(data[data.coherence == 15],depends_on={'v': 'asi_group', 'a':'asi_group'}, std_depends=True, p_outlier=0.10)
va_ASI_15.find_starting_values()
va_ASI_15.sample(20000, burn=2000, thin=5,dbname='va_ASI_15_traces.db', db='pickle')
va_ASI_15.save('va_ASI_15')


va_ASI_15.print_stats()
va_ASI_15=hddm.load('va_ASI_15')

#%%
###########
#HDDM results

v_L, v_H = va_ASI_15.nodes_db.node[['v(low_asi)', 'v(high_asi)']]
hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=8)
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(H<L)="), (v_H.trace()< v_L.trace()).mean()


a_L, a_H = va_ASI_15.nodes_db.node[['a(low_asi)', 'a(high_asi)']]
hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=8)
plt.xlabel('threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions  for ASI')
print ("P(H >L)="),(a_H.trace()> a_L.trace()).mean()




#########
#Get traces of parameters for plotting

va_ASI_traces = va_ASI_15.get_traces()

v_1_asi=va_ASI_traces[['v(low_asi)','v(high_asi)']].rename(columns={'v(low_asi)':'Low ASI','v(high_asi)':'High ASI'})

a_1_asi=va_ASI_traces[['a(low_asi)','a(high_asi)']].rename(columns={'a(low_asi)':'Low ASI','a(high_asi)':'High ASI'})




#################
#Violin plot

#v
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=v_1_asi, palette='flare',linewidth=0.5, inner='point')
ax.set(xlabel ="ASI Groups", ylabel = "Drift-rate")
print ("P(H>L)="), (v_H.trace()> v_L.trace()).mean()


#a
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=a_1_asi, palette='flare',linewidth=0.5, inner='point')
ax.set(xlabel ="ASI Groups", ylabel = "Decision-threshold")
print ("P(H >L)="),(a_H.trace()> a_L.trace()).mean()


#%%

##############################
#v and a X ASI 


va_ASI_25 = hddm.HDDM(data[data.coherence == 25],depends_on={'v': 'asi_group', 'a':'asi_group'}, std_depends=True, p_outlier=0.10)
va_ASI_25.find_starting_values()
va_ASI_25.sample(20000, burn=2000, thin=5,dbname='va_ASI_25_traces.db', db='pickle')
va_ASI_25.save('va_ASI_25')
#%%

va_ASI_25.print_stats()
va_ASI_25=hddm.load('va_ASI_25')


###########
#HDDM results

v_L, v_H = va_ASI_25.nodes_db.node[['v(low_asi)', 'v(high_asi)']]
hddm.analyze.plot_posterior_nodes([v_L, v_H], bins=8)
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(H<L)="), (v_H.trace()< v_L.trace()).mean()


a_L, a_H = va_ASI_25.nodes_db.node[['a(low_asi)', 'a(high_asi)']]
hddm.analyze.plot_posterior_nodes([a_L, a_H], bins=8)
plt.xlabel('threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions  for ASI')
print ("P(H >L)="),(a_H.trace()> a_L.trace()).mean()




#########
#Get traces of parameters for plotting

va_ASI_traces = va_ASI_25.get_traces()

v_1_asi=va_ASI_traces[['v(low_asi)','v(high_asi)']].rename(columns={'v(low_asi)':'Low ASI','v(high_asi)':'High ASI'})

a_1_asi=va_ASI_traces[['a(low_asi)','a(high_asi)']].rename(columns={'a(low_asi)':'Low ASI','a(high_asi)':'High ASI'})




#################
#Violin plot

#v
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=v_1_asi, palette='flare',linewidth=0.5, inner='point')
ax.set(xlabel ="ASI Groups", ylabel = "Drift-rate")
print ("P(H>L)="), (v_H.trace()> v_L.trace()).mean()


#a
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=a_1_asi, palette='flare',linewidth=0.5, inner='point')
ax.set(xlabel ="ASI Groups", ylabel = "Decision-threshold")
print ("P(H >L)="),(a_H.trace()> a_L.trace()).mean()


#%%

################################################
##########################################
####################################
#################################
###########################
#######################
###################
###############
###########
#########
#########
#########
#Figures
va_coh_pdi= hddm.load('va_coh_pdi') 
va_coh_caps= hddm.load('va_coh_caps') 
va_coh_asi= hddm.load('va_coh_asi') 

#Get traces of parameters for plotting

va_coh_pdi_traces = mva_coh_pdi.get_traces()

v_pdi=va_coh_pdi_traces[['v(25.low_pdi)','v(15.low_pdi)',
                          'v(25.high_pdi)','v(15.high_pdi)']].rename(columns={'v(25.low_pdi)':'High precision ','v(15.low_pdi)':'Low precision ',
                                                                                'v(25.high_pdi)':'High precision','v(15.high_pdi)':'Low precision'})

a_pdi=va_coh_pdi_traces[['a(25.low_pdi)','a(15.low_pdi)',
                          'a(25.high_pdi)','a(15.high_pdi)']].rename(columns={'a(25.low_pdi)':'High precision ','a(15.low_pdi)':'Low precision ',
                                                                                'a(25.high_pdi)':'High precision','a(15.high_pdi)':'Low precision'})


v_LPDI_25, v_HPDI_25, v_LPDI_15, v_HPDI_15 = mva_coh_pdi.nodes_db.node[['v(25.low_pdi)', 'v(25.high_pdi)','v(15.low_pdi)', 'v(15.high_pdi)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_25, v_HPDI_25, v_LPDI_15, v_HPDI_15], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_25.trace()< v_HPDI_25.trace()).mean())
print ("P(v HPDI: Sham > v TMS)=",(v_LPDI_15.trace()< v_HPDI_15.trace()).mean())



a_LPDI_25, a_HPDI_25, a_LPDI_15, a_HPDI_15 = mva_coh_pdi.nodes_db.node[['a(25.low_pdi)', 'a(25.high_pdi)','a(15.low_pdi)', 'a(15.high_pdi)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_25, a_HPDI_25, a_LPDI_15, a_HPDI_15 ], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())

#################
#Violin plot

############
##v
fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=v_pdi, palette=sns.diverging_palette(130, 330, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                                     High PDI", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Drift-rate", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(v_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)


# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_25.trace()< v_LPDI_15.trace()).mean())

# statistical annotation
x1, x2 = 2,3   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['High precision'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_25.trace()< a_LPDI_15.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['High precision'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v Sham: High PDI <  Low PDI)=",(v_HPDI_25.trace()< v_LPDI_25.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['High precision'].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v TMS: High PDI >  Low PDI)=",(v_HPDI_15.trace()< v_LPDI_15.trace()).mean())


#############
##a

fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=a_pdi, palette=sns.diverging_palette(130, 330, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                                     High PDI", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Decision-threshold", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(a_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_15.trace()> a_LPDI_25.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['High precision'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_15.trace()> a_HPDI_25.trace()).mean())

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['High precision'].max() + 0.2, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# print ("P(v Sham: High PDI <  Low PDI)=",(a_HPDI_15.trace()< a_LPDI_15.trace()).mean())

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['High precision'].max() + 0.3, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
# print ("P(v TMS: High PDI <  Low PDI)=",(a_HPDI_25.trace()< a_LPDI_25.trace()).mean())


#########
#Get traces of parameters for plotting

va_coh_caps_traces = mva_coh_caps.get_traces()

v_caps=va_coh_caps_traces[['v(low_caps.25)','v(low_caps.15)',
                          'v(high_caps.25)','v(high_caps.15)']].rename(columns={'v(low_caps.25)':'High precision ','v(low_caps.15)':'Low precision ',
                                                                                'v(high_caps.25)':'High precision','v(high_caps.15)':'Low precision'})

a_caps=va_coh_caps_traces[['a(low_caps.25)','a(low_caps.15)',
                          'a(high_caps.25)','a(high_caps.15)']].rename(columns={'a(low_caps.25)':'High precision ','a(low_caps.15)':'Low precision ',
                                                                                'a(high_caps.25)':'High precision','a(high_caps.15)':'Low precision'})


v_LCAPS_25, v_LCAPS_15, v_HCAPS_25, v_HCAPS_15 = mva_coh_caps.nodes_db.node[['v(low_caps.25)','v(low_caps.15)',
                          'v(high_caps.25)','v(high_caps.15)']]


CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0, v_LPDI_1, v_HPDI_0, v_HPDI_1], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')


a_LCAPS_25, a_LCAPS_15, a_HCAPS_25,  a_HCAPS_15 = mva_coh_caps.nodes_db.node[['a(low_caps.25)','a(low_caps.15)',
                          'a(high_caps.25)','a(high_caps.15)']]

CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_25, a_LCAPS_15, a_HCAPS_25,  a_HCAPS_15 ], bins=15)
plt.legend(['a 25 LCAPS','a 15 LCAPS', 'a 25 HCAPS','a 15 HCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('DECISION THRESHOLD')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')

#################
#Violin plot

############
##v

fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=v_caps, palette=sns.diverging_palette(130, 33, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low CAPS                                     High CAPS", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Drift-rate", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(v_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v LPDI: Sham > v TMS)=",(v_LCAPS_25.trace()< v_LCAPS_15.trace()).mean())

# statistical annotation
x1, x2 = 2,3   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['High precision'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v HPDI: Sham > v TMS)=",(v_HCAPS_25.trace()< v_HCAPS_15.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['High precision'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v Sham: High PDI <  Low PDI)=",(v_HCAPS_25.trace() < v_LCAPS_25.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['High precision'].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v TMS: High CAPS >  Low CAPS)=",(v_HCAPS_15.trace() <v_LCAPS_15.trace()).mean())


#############
##a

fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=a_caps, palette=sns.diverging_palette(130, 33, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low CAPS                                     High CAPS", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Decision-threshold", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(a_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a LPDI: Sham > a TMS)=",(a_LCAPS_25.trace()< a_LCAPS_15.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['High precision'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a HPDI: Sham > a TMS)=",(a_HCAPS_25.trace()< a_HCAPS_15.trace()).mean())

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_caps['High precision'].max() + 0.15, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
print ("P(A hp: High CAPS >  Low CAPS)=",(a_HCAPS_25.trace()>  a_LCAPS_25.trace()).mean())

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_caps['High precision'].max() + 0.2, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v TMS: High PDI <  Low PDI)=",(a_HCAPS_15.trace()< a_LCAPS_15.trace()).mean())

#########
#Get traces of parameters for plotting

va_coh_asi_traces = va_coh_asi.get_traces()

v_asi=va_coh_asi_traces[['v(low_asi.25)','v(low_asi.15)',
                          'v(high_asi.25)','v(high_asi.15)']].rename(columns={'v(low_asi.25)':'High precision ','v(low_asi.15)':'Low precision ',
                                                                                'v(high_asi.25)':'High precision','v(high_asi.15)':'Low precision'})

a_asi=va_coh_asi_traces[['a(low_asi.25)','a(low_asi.15)',
                          'a(high_asi.25)','a(high_asi.15)']].rename(columns={'a(low_asi.25)':'High precision ','a(low_asi.15)':'Low precision ',
                                                                                'a(high_asi.25)':'High precision','a(high_asi.15)':'Low precision'})


v_LASI_25, v_LASI_15, v_HASI_25, v_HASI_15 = va_coh_asi.nodes_db.node[['v(low_asi.25)','v(low_asi.15)',
                          'v(high_asi.25)','v(high_asi.15)']]


PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0, v_LPDI_1, v_HPDI_0, v_HPDI_1], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())



a_LASI_25, a_LASI_15, a_HASI_25,  a_HASI_15 = va_coh_asi.nodes_db.node[['a(low_asi.25)','a(low_asi.15)',
                          'a(high_asi.25)','a(high_asi.15)']]


PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_0, a_LPDI_1, a_HPDI_0, a_HPDI_1], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())

#################
#Violin plot
mypalette=sns.diverging_palette(150, 275, s=80, l=55, n=9)


############
##v

fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=v_asi, palette=sns.diverging_palette(130, 230, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low ASI                                     High ASI", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Drift-rate", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(v_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_asi['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v LPDI: Sham > v TMS)=",(v_LASI_25.trace()< v_LASI_15.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_asi['High precision'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v HPDI: Sham > v TMS)=",(v_HASI_25.trace()< v_HASI_15.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_asi['High precision '].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v Sham: High PDI <  Low PDI)=",(v_HASI_25.trace()> v_LASI_25.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_asi['High precision '].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v TMS: High PDI >  Low PDI)=",(v_HASI_15.trace()> v_LASI_15.trace()).mean())


#############
##a

fig, ax = plt.subplots(figsize=(16, 12))
ax = sns.violinplot( data=a_asi, palette=sns.diverging_palette(130, 230, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low ASI                                     High ASI", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Decision-threshold", size=32, weight="bold", labelpad=40)
ax.set_xticklabels(a_pdi, fontsize=20)
ax.tick_params(axis='y', labelsize=20)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_asi['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a LPDI: Sham > a TMS)=",(a_LASI_25.trace()> a_LASI_15.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_asi['High precision '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=20)
print ("P(a HPDI: Sham > a TMS)=",(a_HASI_25.trace()> a_HASI_15.trace()).mean())

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_asi['High precision '].max() + 0.1, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
print ("P(v Sham: High PDI <  Low PDI)=",(a_HASI_25.trace()< a_LASI_25.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_asi['High precision '].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=20)
print ("P(v TMS: High PDI <  Low PDI)=",(a_HASI_15.trace()< a_LASI_15.trace()).mean())



