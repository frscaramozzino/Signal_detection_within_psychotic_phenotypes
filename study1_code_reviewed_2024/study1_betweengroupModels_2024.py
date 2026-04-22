#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:42:32 2024

@author: francescoscaramozzino
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:25:55 2022

@author: PHJT002
"""

import hddm

#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
#%%

#load data and clean outliers as in prereg
data= hddm.load_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 

#Kernel density estimate plot
fig = px.histogram(data, x="rt", color='response',  
                histnorm='density',
                orientation='v',
                opacity=0.4,
                barmode='overlay',
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
nullmodel.sample(20000, burn=2000, thin=5,  dbname='nullmodel_traces.db', db='pickle')
nullmodel.save('nullmodel')

#nullmodel=hddm.load('nullmodel')

nullmodel.print_stats()
nullmodel.plot_posteriors()
# DIC: 16163.188807
# deviance: 16160.157810
# pD: 3.030997
#%%
#  v and a <-- Motion coherence
hddm_c = hddm.HDDM(data,depends_on={'v': 'coherence', 'a':'coherence'}, std_depends=True, p_outlier=0.10)
hddm_c.find_starting_values()
hddm_c.sample(20000, burn=2000, thin=5, dbname='hddm_c_traces.db', db='pickle')
hddm_c.save('hddm_c')

#hddm_c=hddm.load('hddm_c')
hddm_c.print_stats()
hddm_c.plot_posteriors()

# DIC: 16080.556321
# deviance: 16075.500696
# pD: 5.055626

#%%
#  v and a <-- Motion coherence + PDI groups

hddm_c_pdi = hddm.HDDM(data,depends_on={'v': ['pdi_group','coherence'], 
                                                'a':['pdi_group','coherence']}, 
                       std_depends=True, p_outlier=0.05)
hddm_c_pdi.find_starting_values()
hddm_c_pdi.sample(20000, burn=2000, thin=5,  dbname='hddm_c_pdi_traces.db', db='pickle')
hddm_c_pdi.save('hddm_c_pdi')

hddm_c_pdi= hddm.load('hddm_c_pdi') 

hddm_c_pdi.print_stats()

# DIC: 16188.751547
# deviance: 16179.826599
# pD: 8.924948

#%%

#  v and a <-- Motion coherence + CAPS groups

hddm_c_caps = hddm.HDDM(data,depends_on={'v': ['caps_group','coherence'], 
                                                'a':['caps_group','coherence']}, std_depends=True, p_outlier=0.10)
hddm_c_caps.find_starting_values()
hddm_c_caps.sample(20000, burn=2000, thin=5,  dbname='hddm_c_caps_traces.db', db='pickle')
hddm_c_caps.save('hddm_c_caps')


hddm_c_caps= hddm.load('hddm_c_caps') 
hddm_c_caps.print_stats()
# DIC: 16067.085984
# deviance: 16058.178657
# pD: 8.907327
#%%

#  v and a <-- Motion coherence + ASI groups

hddm_c_asi = hddm.HDDM(data,depends_on={'v': ['asi_group','coherence'], 
                                                'a':['asi_group','coherence']}, std_depends=True, p_outlier=0.10)
hddm_c_asi.find_starting_values()
hddm_c_asi.sample(20000, burn=2000, thin=5,  dbname='hddm_c_asi_traces.db', db='pickle')
hddm_c_asi.save('hddm_c_asi')

hddm_c_asi= hddm.load('hddm_c_asi') 
hddm_c_asi.print_stats()

# DIC: 16078.927262
# deviance: 16069.967960
# pD: 8.959302
#%%

#Comparison parameter distributions 

v_L15, v_H15 = hddm_c_pdi.nodes_db.node[['v(15.low_pdi)', 'v(15.high_pdi)']]
print ("v15: Hpdi < Lpdi="), (v_H15.trace()< v_L15.trace()).mean()


v_L25, v_H25 = hddm_c_pdi.nodes_db.node[['v(25.low_pdi)', 'v(25.high_pdi)']]
print ("v25: Hpdi < Lpdi="), (v_H25.trace()< v_L25.trace()).mean()


a_L15, a_H15 = hddm_c_pdi.nodes_db.node[['a(15.low_pdi', 'a(15.high_pdi)']]
print ("a15: Hpdi > Lpdi="),(a_H15.trace()> a_L15.trace()).mean()


a_L25, a_H25 = hddm_c_pdi.nodes_db.node[['a(25.low_pdi)', 'a(25.high_pdi)']]
print ("a25: Hpdi > Lpdi="),(a_H25.trace()>a_L25.trace()).mean()


#%%

#Comparison parameter distributions 

v_L15, v_H15 = hddm_c_caps.nodes_db.node[['v(low_caps.15)', 'v(high_caps.15)']]
print ("v15: HCAPS < LCAPS="), (v_H15.trace()< v_L15.trace()).mean()


v_L25, v_H25 = hddm_c_caps.nodes_db.node[['v(low_caps.25)', 'v(high_caps.25)']]
print ("v25: HCAPS < LCAPS="), (v_H25.trace()< v_L25.trace()).mean()


a_L15, a_H15 = hddm_c_caps.nodes_db.node[['a(low_caps.15)', 'a(high_caps.15)']]
print ("a15: HCAPS > LCAPS="),(a_H15.trace()> a_L15.trace()).mean()


a_L25, a_H25 = hddm_c_caps.nodes_db.node[['a(low_caps.25)', 'a(high_caps.25)']]
print ("a25: HCAPS > LCAPS="),(a_H25.trace()>a_L25.trace()).mean()

#%%
#Comparison parameter distributions 

v_L15, v_H15 = hddm_c_asi.nodes_db.node[['v(low_asi.15)', 'v(high_asi.15)']]
print ("v15: HASI < LASI="), (v_H15.trace()< v_L15.trace()).mean()


v_L25, v_H25 = hddm_c_asi.nodes_db.node[['v(low_asi.25)', 'v(high_asi.25)']]
print ("v25: HASI > LASI="), (v_H25.trace()> v_L25.trace()).mean()


a_L15, a_H15 = hddm_c_asi.nodes_db.node[['a(low_asi.15)', 'a(high_asi.15)']]
print ("a15: HASI > LASI="),(a_H15.trace()> a_L15.trace()).mean()


a_L25, a_H25 = hddm_c_asi.nodes_db.node[['a(low_asi.25)', 'a(high_asi.25)']]
print ("a25: HASI > LASI="),(a_H25.trace()>a_L25.trace()).mean()











