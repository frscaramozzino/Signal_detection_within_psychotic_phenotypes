#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:48:53 2023

@author: francescoscaramozzino
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import hddm


data= pd.read_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 
#%%


# Group by Participant and Condition, then calculate mean for RT and Accuracy
mean_df = data.groupby(['sub_idx', 'coherence']).agg(
    RT=('rt', 'mean'),
    Accuracy=('response', 'mean')
).reset_index()

# Pivot table to get a wide format for each condition (e.g. RT_condition1, RT_condition2)
pivot_df = mean_df.pivot(index='sub_idx', columns='coherence', values=['RT', 'Accuracy'])

# Rename the columns to be more readable
pivot_df.columns = ['RT_LC', 'RT_HC', 'Accuracy_LC', 'Accuracy_HC']

# Calculate the overall RT and Accuracy by grouping by Participant and taking the mean
data_corr=data.groupby(['sub_idx'])['age','PDI', 'caps', 'asi', 'crt','beadsDraws','ipr','pr','la_bead','overAdjust','rt','response'].mean()

data_corr=data_corr.rename(columns={'age':'Age','crt':'CRT', 'caps':'CAPS', 'asi':'ASI','beadsDraws':'DTD', 'ipr':'confidence_1st','pr':'prior_confidence', 'la_bead':'confidence_final', 'overAdjust':'ADJ','rt':'RT','response':'Accuracy'})

# Merge overall means into the pivoted dataframe
final_df = pd.concat([data_corr,pivot_df], axis=1).reset_index()


data_glm=(final_df-final_df.mean())/final_df.std()


#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:22:29 2021

@author: francescoscaramozzino
"""





#Forest plot
col=['coef','y']
col1=['coef','[0.025','y']
col2=['coef','0.975]','y']


list_of_responses = ['RT',
                         'RT_HC',
                         'RT_LC',
                         'Accuracy',
                         'Accuracy_HC',
                         'Accuracy_LC',
                         'DTD',
                         'prior_confidence',
                         'confidence_1st',
                         'confidence_final',
                         'ADJ']

# list of models
models = []
res_models = []
df_list=[]

for resp in list_of_responses:
    formula = resp + " ~ PDI"
    models.append(smf.quantreg(formula, data_glm).fit(q=.5))


for mod in models:
    res_models.append(pd.read_html(mod.summary().tables[1].as_html(),
                                         header=0,index_col=0)[0])


res_RT =res_models[0].assign(y= "RT").loc[['PDI']]
lb=res_RT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)

res_RT=pd.concat([lb,hb])

res_RT_HC =res_models[1].assign(y= "RT_HC").loc[['PDI']]
lb=res_RT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_HC=pd.concat([lb,hb])

res_RT_LC=res_models[2].assign(y= "RT_LC").loc[['PDI']]
lb=res_RT_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_LC=pd.concat([lb,hb])

res_Accuracy =res_models[3].assign(y= "Accuracy").loc[['PDI']]
lb=res_Accuracy[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy=pd.concat([lb,hb])

res_Accuracy_HC =res_models[4].assign(y= "Accuracy_HC").loc[['PDI']]
lb=res_Accuracy_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_HC=pd.concat([lb,hb])

res_Accuracy_LC =res_models[5].assign(y= "Accuracy_LC").loc[['PDI']]
lb=res_Accuracy_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_LC=pd.concat([lb,hb])

res_DTD =res_models[6].assign(y= "DTD").loc[['PDI']]
lb=res_DTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_DTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_DTD=pd.concat([lb,hb])

res_prior_confidence =res_models[7].assign(y= "prior_confidence").loc[['PDI']]
lb=res_prior_confidence[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_prior_confidence[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_prior_confidence=pd.concat([lb,hb])

res_confidence_1st =res_models[8].assign(y= "confidence_1st").loc[['PDI']]
lb=res_confidence_1st[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_1st[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_1st=pd.concat([lb,hb])

res_confidence_final =res_models[9].assign(y= "confidence_final").loc[['PDI']]
lb=res_confidence_final[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_final[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_final=pd.concat([lb,hb])

res_ADJ =res_models[10].assign(y= "ADJ").loc[['PDI']]
lb=res_ADJ[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_ADJ[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_ADJ=pd.concat([lb,hb])



res_pdi=pd.concat([res_RT,
                   res_RT_HC,
                   res_RT_LC,
                   res_Accuracy,
                   res_Accuracy_HC,res_Accuracy_LC,
                   res_DTD, 
                   res_prior_confidence,
                   res_confidence_1st, 
                   res_confidence_final,
                   res_ADJ])
res_pdi=res_pdi.assign( Schizotipy = "PDI")
#%%


formula = " confidence_final~ PDI+DTD"
result= smf.quantreg(formula, data_glm).fit(q=.5)
print(result.summary())

formula = " confidence_final~ CAPS+DTD"
result= smf.quantreg(formula, data_glm).fit(q=.5)
print(result.summary())

formula = " confidence_final~ ASI+DTD"
result= smf.quantreg(formula, data_glm).fit(q=.5)
print(result.summary())

#%%
####CAPS
# list of models
models = []
res_models = []

for resp in list_of_responses:
    formula = resp + " ~ CAPS"
    models.append(smf.quantreg(formula, data_glm).fit(q=.5))


for mod in models:
    res_models.append(pd.read_html(mod.summary().tables[1].as_html(),
                                         header=0,index_col=0)[0])



res_RT =res_models[0].assign(y= "RT").loc[['CAPS']]
lb=res_RT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT=pd.concat([lb,hb])

res_RT_HC =res_models[1].assign(y= "RT_HC").loc[['CAPS']]
lb=res_RT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_HC=pd.concat([lb,hb])

res_RT_LC=res_models[2].assign(y= "RT_LC").loc[['CAPS']]
lb=res_RT_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_LC=pd.concat([lb,hb])

res_Accuracy =res_models[3].assign(y= "Accuracy").loc[['CAPS']]
lb=res_Accuracy[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy=pd.concat([lb,hb])

res_Accuracy_HC =res_models[4].assign(y= "Accuracy_HC").loc[['CAPS']]
lb=res_Accuracy_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_HC=pd.concat([lb,hb])

res_Accuracy_LC =res_models[5].assign(y= "Accuracy_LC").loc[['CAPS']]
lb=res_Accuracy_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_LC=pd.concat([lb,hb])

res_DTD =res_models[6].assign(y= "DTD").loc[['CAPS']]
lb=res_DTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_DTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_DTD=pd.concat([lb,hb])

res_prior_confidence =res_models[7].assign(y= "prior_confidence").loc[['CAPS']]
lb=res_prior_confidence[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_prior_confidence[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_prior_confidence=pd.concat([lb,hb])

res_confidence_1st =res_models[8].assign(y= "confidence_1st").loc[['CAPS']]
lb=res_confidence_1st[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_1st[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_1st=pd.concat([lb,hb])

res_confidence_final =res_models[9].assign(y= "confidence_final").loc[['CAPS']]
lb=res_confidence_final[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_final[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_final=pd.concat([lb,hb])

res_ADJ =res_models[10].assign(y= "ADJ").loc[['CAPS']]
lb=res_ADJ[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_ADJ[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_ADJ=pd.concat([lb,hb])


res_caps=pd.concat([res_RT,
                   res_RT_HC,
                   res_RT_LC,
                   res_Accuracy,
                   res_Accuracy_HC,res_Accuracy_LC,
                   res_DTD, 
                   res_prior_confidence,
                   res_confidence_1st, 
                   res_confidence_final,
                   res_ADJ])
res_caps=res_caps.assign( Schizotipy = "CAPS")

#%%
####ASI
# list of models
models = []
res_models = []

for resp in list_of_responses:
    formula = resp + " ~ ASI"
    models.append(smf.quantreg(formula, data_glm).fit(q=.5))


for mod in models:
    res_models.append(pd.read_html(mod.summary().tables[1].as_html(),
                                         header=0,index_col=0)[0])



res_RT =res_models[0].assign(y= "RT").loc[['ASI']]
lb=res_RT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT=pd.concat([lb,hb])

res_RT_HC =res_models[1].assign(y= "RT_HC").loc[['ASI']]
lb=res_RT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_HC=pd.concat([lb,hb])

res_RT_LC=res_models[2].assign(y= "RT_LC").loc[['ASI']]
lb=res_RT_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_RT_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_RT_LC=pd.concat([lb,hb])

res_Accuracy =res_models[3].assign(y= "Accuracy").loc[['ASI']]
lb=res_Accuracy[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy=pd.concat([lb,hb])

res_Accuracy_HC =res_models[4].assign(y= "Accuracy_HC").loc[['ASI']]
lb=res_Accuracy_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_HC=pd.concat([lb,hb])

res_Accuracy_LC =res_models[5].assign(y= "Accuracy_LC").loc[['ASI']]
lb=res_Accuracy_LC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_Accuracy_LC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_Accuracy_LC=pd.concat([lb,hb])

res_DTD =res_models[6].assign(y= "DTD").loc[['ASI']]
lb=res_DTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_DTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_DTD=pd.concat([lb,hb])

res_prior_confidence =res_models[7].assign(y= "prior_confidence").loc[['ASI']]
lb=res_prior_confidence[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_prior_confidence[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_prior_confidence=pd.concat([lb,hb])

res_confidence_1st =res_models[8].assign(y= "confidence_1st").loc[['ASI']]
lb=res_confidence_1st[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_1st[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_1st=pd.concat([lb,hb])

res_confidence_final =res_models[9].assign(y= "confidence_final").loc[['ASI']]
lb=res_confidence_final[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_confidence_final[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_confidence_final=pd.concat([lb,hb])

res_ADJ =res_models[10].assign(y= "ADJ").loc[['ASI']]
lb=res_ADJ[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_ADJ[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_ADJ=pd.concat([lb,hb])


res_asi=pd.concat([res_RT,
                   res_RT_HC,
                   res_RT_LC,
                   res_Accuracy,
                   res_Accuracy_HC,res_Accuracy_LC,
                   res_DTD, 
                   res_prior_confidence,
                   res_confidence_1st, 
                   res_confidence_final,
                   res_ADJ])  
    
res_asi=res_asi.assign( Schizotipy = "ASI")

res_comp=pd.concat([res_pdi, res_caps, res_asi])

#%%

res_comp=res_comp.replace({'RT':'Mean RT',
                         'RT_HC':'Mean HC RT',
                         'RT_LC':'Mean LC RT',
                         'Accuracy':'Mean Accuracy',
                         'Accuracy_HC':'Mean HC Accuracy',
                         'Accuracy_LC':'Mean LC Accuracy',
                         'prior_confidence':'Conf-0',
                         'confidence_1st':'Conf-1',
                         'confidence_final':'Conf-N',
                         'ADJ':'ADJ'})

res_comp.to_csv('study1_results_qr_2024.csv')

#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

res_comp=pd.read_csv('study1_results_qr_2024.csv')
# res_comp = res_comp.sort_values(by='coef', ascending=True)

res_comp.set_index('Unnamed: 0', inplace=True)



#%%
list(data_glm.columns.values.tolist())

coloursdist={
         "PDI": "rgb(135, 197, 95)",
         "CAPS": "rgb(248, 156, 116)",
         "ASI": "rgb(180, 151, 231)"}

fig = px.scatter(res_comp, 
                 x="b", 
                 y="y",
                 color=res_comp.index,
                 color_discrete_map=coloursdist, 
                 facet_col=res_comp.Schizotipy,
                 labels= {'index':'Psychometric',
                         'b':'β', 'Unnamed: 0':'Psychometric'
                         },
                 width=1500,
                 height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


# iterate on each y
for i in res_comp["y"].unique():
    # filter by y
    df_sub = res_comp[res_comp["y"] == i]
    
    fig.add_shape(
        type="line",
        layer="below",
         xref='x1',line=dict(color=coloursdist['PDI'],width=7),
        # connect the two markers
        ## e.g., y0='Robredo', x0=43.53
        y0=df_sub.y.values[0], x0=df_sub.loc['PDI','b'].min(),
        ## e.g., y1='Marcos', x1=26.60
        y1=df_sub.y.values[1], x1=df_sub.loc['PDI','b'].max(),
    )
    fig.add_annotation(
     y=df_sub.y.values[1], x=df_sub.coef.values[1],
     text='|',
     showarrow=False,
     font=dict(size=22, color="black", family="Courier New, monospace"),
     )

    fig.add_shape(
        type="line",
        layer="below",
         xref='x2',line=dict(color=coloursdist['CAPS'],width=7),
        # connect the two markers
        ## e.g., y0='Robredo', x0=43.53
        y0=df_sub.y.values[0], x0=df_sub.loc['CAPS','b'].min(),
        ## e.g., y1='Marcos', x1=26.60
        y1=df_sub.y.values[1], x1=df_sub.loc['CAPS','b'].max(),
    )
    fig.add_annotation(
     y=df_sub.y.values[1], x=df_sub.loc['CAPS','coef'].values[1],
     text='|',xref='x2',
     showarrow=False,
     font=dict(size=22, color="black", family="Courier New, monospace"),
     )
  
    fig.add_shape(
        type="line",
        layer="below",
         xref='x3',line=dict(color=coloursdist['ASI'],width=7),
        # connect the two markers
        ## e.g., y0='Robredo', x0=43.53
        y0=df_sub.y.values[0], x0=df_sub.loc['ASI','b'].min(),
        ## e.g., y1='Marcos', x1=26.60
        y1=df_sub.y.values[1], x1=df_sub.loc['ASI','b'].max(),
    )
    fig.add_annotation(
     y=df_sub.y.values[1], x=df_sub.loc['ASI','coef'].values[1],
     text='|', xref='x3',
     showarrow=False,
     font=dict(size=22, color="black", family="Courier New, monospace"),
     )

fig.add_annotation(
      y='Conf-N', x=0.5,
      text='<b>**<b>',
      showarrow=False,
      font=dict(size=40, color="black", family="Courier New, monospace"),
      xref='x1',)    

fig.add_annotation(
      y='Conf-N', x=0.5,
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=40, color="black", family="Courier New, monospace"),
      xref='x2',)    

fig.add_annotation(
      y='Conf-N', x=0.5,
      text='<b>**<b>',
      showarrow=False,
      font=dict(size=40, color="black", family="Courier New, monospace"),
      xref='x3',)    

fig.add_annotation(
      y=1.055, x=-0.02,
      text='<b>-<b>',
      showarrow=False,
      font=dict(size=100, color=coloursdist['PDI'], family="Courier New, monospace"),
      xref='x1',
      yref='paper') 
fig.add_annotation(
      y=1.055, x=-0.05,
      text='<b>-<b>',
      showarrow=False,
      font=dict(size=100, color=coloursdist['CAPS'], family="Courier New, monospace"),
      xref='x2',
      yref='paper') 

fig.add_annotation(
      y=1.055, x=-0.02,
      text='<b>-<b>',
      showarrow=False,
      font=dict(size=100, color=coloursdist['ASI'], family="Courier New, monospace"),
      xref='x3',
      yref='paper')    


fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='line-ne', size=15)
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    name="CAPS",
    text='<b>**<b>',    
))
# Add the vertical lines at x=0 in each column, with extended y limits
for facet in range(1, 4):  # Assuming 3 facet columns
    fig.add_shape(
        type="line",
        x0=0, x1=0,  # x=0 for vertical line
        y0=11, y1=-1,  # Extended y values
        xref=f'x{facet}',  # x1, x2, x3 depending on the facet column
        yref=f'y{facet}',
        line=dict(color="grey", width=2)  # Black line
    )


fig.show()



fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=40,
    # title_font_family="Times New Roman",
    # title_font_color="black",
    # title="B.",
    legend_title_text=None,
    legend_font_size=40,
    legend=dict(itemsizing="constant",
                itemwidth=80),
    width=1900,  # Set your desired width in pixels
    height=1700  # Set your desired height in pixels
)
fig.update_layout(showlegend=False)

fig.show()

fig.write_image("study1_fig_qr_B_2024.png",scale=3)

