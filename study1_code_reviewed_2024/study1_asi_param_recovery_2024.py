#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:12:10 2024

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:26:24 2024

@author: francescoscaramozzino
"""



import hddm
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from tableone import TableOne, load_dataset
from scipy.special import rel_entr
import random

#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


data= hddm.load_csv('data_study1.csv')
data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 



#Create simulated data for parameter recovery

mva_coh_asi= hddm.load('hddm_c_asi') 
mva_coh_asi.print_stats()
tms_s=mva_coh_asi.gen_stats()
tms_s=tms_s.T
#%%

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

#%%
###Running model with simulated data
vat_asi_coh_sim2 = hddm.HDDM(data_sim,depends_on={'v':['asi_group','coherence'], 
                                                  'a':['asi_group','coherence']},
                        std_depends=True, p_outlier=0.10)
vat_asi_coh_sim2.find_starting_values()
vat_asi_coh_sim2.sample(20000, burn=2000, thin=5,  dbname='hddm_c_asi_sim.db', db='pickle')
vat_asi_coh_sim2.save('hddm_c_asi_sim')

vat_asi_coh_sim2= hddm.load('hddm_c_asi_sim')


vat_asi_coh_sim2.print_stats()


vat_asi_coh_sim_traces = vat_asi_coh_sim2.get_traces()
mva_coh_asi_traces = mva_coh_asi.get_traces()
mva_coh_asi.print_stats()
#%%

############################### DRIFT RATE

v_15_0, v_15_1, v_5_0, v_5_1 = mva_coh_asi.nodes_db.node[['v(low_asi.15)', 'v(high_asi.15)',
                                                          'v(low_asi.25)', 'v(high_asi.25)']]

vs_15_0, vs_15_1, vs_5_0, vs_5_1 = vat_asi_coh_sim2.nodes_db.node[['v(low_asi.15)', 'v(high_asi.15)',
                                                          'v(low_asi.25)', 'v(high_asi.25)']]


vposteriors= hddm.analyze.plot_posterior_nodes([v_15_0, v_15_1, v_5_0, v_5_1], bins=15)
plt.legend(['Low ASI-Low Precision','High ASI-Low Precision ','Low ASI-High Precision','High ASI-High Precision '],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(v Low Precision: Sham >  TMS)=",(v_15_0.trace()< v_15_1.trace()).mean())
print ("P(v High Precision: Sham >  TMS)=",(v_5_0.trace()< v_5_1.trace()).mean())



vposteriors= hddm.analyze.plot_posterior_nodes([vs_15_0, vs_15_1, vs_5_0, vs_5_1], bins=15)
plt.legend(['Low ASI-Low Precision','High ASI-Low Precision ','Low ASI-High Precision','High ASI-High Precision '],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(v Low Precision: Sham >  TMS)=",(vs_15_0.trace()< vs_15_1.trace()).mean())
print ("P(v High Precision: Sham >  TMS)=",(vs_5_0.trace()< vs_5_1.trace()).mean())


#%%

############################### Decision Threshold

a_15_0, a_15_1, a_5_0, a_5_1 = mva_coh_asi.nodes_db.node[['a(low_asi.15)', 'a(high_asi.15)',
                                                          'a(low_asi.25)', 'a(high_asi.25)']]

as_15_0, as_15_1, as_5_0, as_5_1 = vat_asi_coh_sim2.nodes_db.node[['a(low_asi.15)', 'a(high_asi.15)',
                                                          'a(low_asi.25)', 'a(high_asi.25)']]


aposteriors= hddm.analyze.plot_posterior_nodes([a_15_0, a_15_1, a_5_0, a_5_1], bins=15)
plt.legend(['Low ASI-Low Precision','High ASI-Low Precision ','Low ASI-High Precision','High ASI-High Precision '],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(a Low Precision: Sham >  TMS)=",(a_15_0.trace()< a_15_1.trace()).mean())
print ("P(a High Precision: Sham >  TMS)=",(a_5_0.trace()< a_5_1.trace()).mean())



aposteriors= hddm.analyze.plot_posterior_nodes([as_15_0, as_15_1, as_5_0, as_5_1], bins=15)
plt.legend(['Low ASI-Low Precision','High ASI-Low Precision ','Low ASI-High Precision','High ASI-High Precision '],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for ASI')
print ("P(a Low Precision: Sham >  TMS)=",(as_15_0.trace()< as_15_1.trace()).mean())
print ("P(a High Precision: Sham >  TMS)=",(as_5_0.trace()< as_5_1.trace()).mean())

