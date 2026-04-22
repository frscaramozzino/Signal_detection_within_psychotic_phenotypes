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

# grouping as suggested by Reviewer 1 as in Duhamel et al 2023

# Compute 81st percentile
caps_80 = np.percentile(data['caps'], 80)
print(caps_80)

pdi_80 = np.percentile(data['PDI'], 80)
print(pdi_80)


# Drop rows where 0 < caps < caps_81
data = data.drop(data[(data['caps'] > 0) & (data['caps'] < caps_80)].index)

# Assign group labels
data['caps_group'] = np.where(data['caps'] == 0, 'low_caps', 'high_caps')

n_low_caps= data[data['caps_group'] == 'low_caps']
n_low_caps=len(pd.unique(n_low_caps['sub_idx']))
print(n_low_caps)

n_high_caps= data[data['caps_group'] == 'high_caps']
n_high_caps=len(pd.unique(n_high_caps['sub_idx']))
print(n_high_caps)

#%%


#Kernel density estimate plot
fig = px.histogram(data, x="rt", color='response',  
                histnorm='density',
                orientation='v',
                opacity=0.4,
                width=1733,
                height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=20, 
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")

fig.show()

#%%
# Null model
nullmodel= hddm.HDDM(data, p_outlier=0.10)
nullmodel.find_starting_values()
nullmodel.sample(20000, burn=2000, thin=5,  dbname='nullmodel_reviewer1_traces.db', db='pickle')
nullmodel.save('nullmodel_reviewer1')

#nullmodel=hddm.load('nullmodel')

nullmodel.print_stats() 
#nullmodel.plot_posteriors()

# DIC: 7944.370652
# deviance: 7941.330232
# pD: 3.040420
#%%
#  v and a X coherence
hddm_c = hddm.HDDM(data,depends_on={'v': 'coherence', 'a':'coherence'}, std_depends=True, p_outlier=0.10)
hddm_c.find_starting_values()
hddm_c.sample(20000, burn=2000, thin=5, dbname='hddm_c_traces_reviewer1.db', db='pickle')
hddm_c.save('hddm_c_reviewer1')

#hddm_c=hddm.load('hddm_c')
hddm_c.print_stats()
#hddm_c.plot_posteriors()

# DIC: 7904.078491
# deviance: 7899.222937
# pD: 4.855554

#%%
hddm_c_caps = hddm.HDDM(data,depends_on={'v': ['caps_group','coherence'], 
                                                'a':['caps_group','coherence']}, std_depends=True, p_outlier=0.10)
hddm_c_caps.find_starting_values()
hddm_c_caps.sample(20000, burn=2000, thin=5,  dbname='hddm_c_caps_traces_reviewer1.db', db='pickle')
hddm_c_caps.save('hddm_c_caps_reviewer1')


hddm_c_caps= hddm.load('hddm_c_caps_reviewer1') 
hddm_c_caps.print_stats()

# DIC: 7888.766122
# deviance: 7879.697779
# pD: 9.068343




#%%

#hddm_c_caps winning over Null and C models

v_L15, v_H15 = hddm_c_caps.nodes_db.node[['v(low_caps.15)', 'v(high_caps.15)']]
print (("v15: HCAPS < LCAPS="), (v_H15.trace()< v_L15.trace()).mean())


v_L25, v_H25 = hddm_c_caps.nodes_db.node[['v(low_caps.25)', 'v(high_caps.25)']]
print (("v25: HCAPS < LCAPS="), (v_H25.trace()< v_L25.trace()).mean())


a_L15, a_H15 = hddm_c_caps.nodes_db.node[['a(low_caps.15)', 'a(high_caps.15)']]
print (("a15: HCAPS > LCAPS="),(a_H15.trace()> a_L15.trace()).mean())


a_L25, a_H25 = hddm_c_caps.nodes_db.node[['a(low_caps.25)', 'a(high_caps.25)']]
print (("a25: HCAPS > LCAPS="),(a_H25.trace()>a_L25.trace()).mean())

#%%
#hddm_c_asi winning over Null and C models

v_L15, v_H15 = hddm_c_asi.nodes_db.node[['v(low_asi.15)', 'v(high_asi.15)']]
print ("v15: HASI < LASI="), (v_H15.trace()< v_L15.trace()).mean()


v_L25, v_H25 = hddm_c_asi.nodes_db.node[['v(low_asi.25)', 'v(high_asi.25)']]
print ("v25: HASI > LASI="), (v_H25.trace()> v_L25.trace()).mean()


a_L15, a_H15 = hddm_c_asi.nodes_db.node[['a(low_asi.15)', 'a(high_asi.15)']]
print ("a15: HASI > LASI="),(a_H15.trace()> a_L15.trace()).mean()


a_L25, a_H25 = hddm_c_asi.nodes_db.node[['a(low_asi.25)', 'a(high_asi.25)']]
print ("a25: HASI > LASI="),(a_H25.trace()>a_L25.trace()).mean()


#%%
data= hddm.load_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

# grouping as suggested by Reviewer 1 as in Duhamel et al 2023

# Compute 81st percentile

pdi_80 = np.percentile(data['PDI'], 80)
print(pdi_80)


# Drop rows where 0 < caps < caps_81

#pdi
data = data.drop(data[(data['PDI'] > 0) & (data['PDI'] < pdi_80)].index)

# Assign group labels
data['pdi_group'] = np.where(data['PDI'] == 0, 'low_pdi', 'high_pdi')

n_low= data[data['pdi_group'] == 'low_pdi']
n_low=len(pd.unique(n_low['sub_idx']))
print(n_low)

n_high= data[data['pdi_group'] == 'high_pdi']
n_high=len(pd.unique(n_high['sub_idx']))
print(n_high)

#%%


#Kernel density estimate plot
fig = px.histogram(data, x="rt", color='response',  
                histnorm='density',
                orientation='v',
                opacity=0.4,
                width=1733,
                height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=20, 
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")

fig.show()

#%%
# Null model
nullmodel= hddm.HDDM(data, p_outlier=0.10)
nullmodel.find_starting_values()
nullmodel.sample(20000, burn=2000, thin=5,  dbname='nullmodel_reviewer1_traces_pdi.db', db='pickle')
nullmodel.save('nullmodel_reviewer1_pdi')

#nullmodel=hddm.load('nullmodel')

nullmodel.print_stats() 
#nullmodel.plot_posteriors()

# DIC: 6806.330802
# deviance: 6803.361881
# pD: 2.968922
#%%
#  v and a X coherence
hddm_c = hddm.HDDM(data,depends_on={'v': 'coherence', 'a':'coherence'}, std_depends=True, p_outlier=0.10)
hddm_c.find_starting_values()
hddm_c.sample(20000, burn=2000, thin=5, dbname='hddm_c_traces_reviewer1_pdi.db', db='pickle')
hddm_c.save('hddm_c_reviewer1_pdi')

#hddm_c=hddm.load('hddm_c')
hddm_c.print_stats()
#hddm_c.plot_posteriors()

# DIC: 6759.340032
# deviance: 6754.332526
# pD: 5.007506
#%%
hddm_c_pdi = hddm.HDDM(data,depends_on={'v': ['pdi_group','coherence'], 
                                                'a':['pdi_group','coherence']}, 
                       std_depends=True, p_outlier=0.10)
hddm_c_pdi.find_starting_values()
hddm_c_pdi.sample(20000, burn=2000, thin=5,  dbname='hddm_c_pdi_traces_reviewer1_pdi.db', db='pickle')
hddm_c_pdi.save('hddm_c_pdi_reviewer1_pdi')

#hddm_c_pdi= hddm.load('hddm_c_pdi_reviewer1_pdi') 

hddm_c_pdi.print_stats()

# DIC: 6760.291917
# deviance: 6751.382388
# pD: 8.909529




#%%

data= hddm.load_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

# grouping as suggested by Reviewer 1 as in Duhamel et al 2023

# Compute 81st percentile
asi_80 = np.percentile(data['asi'], 80)
print(asi_80)


# Drop rows where 0 < caps < caps_81
data = data.drop(data[(data['asi'] > 0) & (data['asi'] < asi_80)].index)

# Assign group labels
data['asi_group'] = np.where(data['asi'] == 0, 'low_asi', 'high_asi')

n_low= data[data['asi_group'] == 'low_asi']
n_low=len(pd.unique(n_low['sub_idx']))
print(n_low)

n_high= data[data['asi_group'] == 'high_asi']
n_high=len(pd.unique(n_high['sub_idx']))
print(n_high)

#%%
# Null model
nullmodel= hddm.HDDM(data, p_outlier=0.10)
nullmodel.find_starting_values()
nullmodel.sample(20000, burn=2000, thin=5,  dbname='nullmodel_reviewer1_traces_asi.db', db='pickle')
nullmodel.save('nullmodel_reviewer1_asi')

#nullmodel=hddm.load('nullmodel')

nullmodel.print_stats() 
#nullmodel.plot_posteriors()

# DIC: 6044.321682
# deviance: 6041.284319
# pD: 3.037363
#%%
#  v and a X coherence
hddm_c = hddm.HDDM(data,depends_on={'v': 'coherence', 'a':'coherence'}, std_depends=True, p_outlier=0.10)
hddm_c.find_starting_values()
hddm_c.sample(20000, burn=2000, thin=5, dbname='hddm_c_traces_reviewer1_asi.db', db='pickle')
hddm_c.save('hddm_c_reviewer1_asi')

#hddm_c=hddm.load('hddm_c')
hddm_c.print_stats()
#hddm_c.plot_posteriors()

# DIC: 6019.491732
# deviance: 6014.572933
# pD: 4.918799
#%%
hddm_c_asi = hddm.HDDM(data,depends_on={'v': ['asi_group','coherence'], 
                                                'a':['asi_group','coherence']}, std_depends=True, p_outlier=0.10)
hddm_c_asi.find_starting_values()
hddm_c_asi.sample(20000, burn=2000, thin=5,  dbname='hddm_c_asi_traces_reviewer1.db', db='pickle')
hddm_c_asi.save('hddm_c_asi_reviewer1')

#hddm_c_asi= hddm.load('hddm_c_asi_reviewer1') 
hddm_c_asi.print_stats()

# DIC: 6025.208874
# deviance: 6016.175994
# pD: 9.032880

