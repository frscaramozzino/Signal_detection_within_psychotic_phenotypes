#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:56:54 2022

@author: francescoscaramozzino
"""
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import numpy as np 
import statsmodels.api as sm

data= pd.read_csv('data_study1.csv')

data = data[(data['rt'] > 0.3)] 
data = data[(data['rt'] < 6)] 
#%%


# Group by Participant and Condition, then calculate mean for RT and Accuracy
mean_df = data.groupby(['sub_idx', 'coherence']).agg(
    mean_RT=('rt', 'mean'),
    Accuracy=('response', 'mean')
).reset_index()

# Pivot table to get a wide format for each condition (e.g. RT_condition1, RT_condition2)
pivot_df = mean_df.pivot(index='sub_idx', columns='coherence', values=['mean_RT', 'Accuracy'])

# Rename the columns to be more readable
pivot_df.columns = ['RT_LC', 'RT_HC', 'Accuracy_LC', 'Accuracy_HC']

# Calculate the overall RT and Accuracy by grouping by Participant and taking the mean
data_corr=data.groupby(['sub_idx'])['age','PDI', 'caps', 'asi', 'crt','beadsDraws','ipr','pr','la_bead','overAdjust','rt','response'].mean()

data_corr=data_corr.rename(columns={'age':'Age','crt':'CRT', 'caps':'CAPS', 'asi':'ASI','beadsDraws':'DTD', 'ipr':'Conf-1','pr':'Conf-0', 'la_bead':'Conf-N', 'overAdjust':'ADJ','rt':'RT','response':'Accuracy'})

# Merge overall means into the pivoted dataframe
final_df = pd.concat([data_corr,pivot_df], axis=1).reset_index()


#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:22:29 2021

@author: francescoscaramozzino
"""



data_glm=(final_df-final_df.mean())/final_df.std()

### GLM PDI 
pdi_c= sm.add_constant(data_glm["PDI"], prepend=False)



#Forest plot
col=['coef','y']
col1=['coef','[0.025','y']
col2=['coef','0.975]','y']

#%%
from statsmodels import graphics

###########PDI
# DTD

glm_PDIxDTD = sm.GLM(data_glm["DTD"], pdi_c)
res_glm_PDIxDTD = glm_PDIxDTD.fit()

#plot residuals
fig, ax = plt.subplots()

resid = res_glm_PDIxDTD.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');

graphics.gofplots.qqplot(resid, line='r')

#produce regression plots
# figsz = plt.figure(figsize=(12,8))
# fig_mod_dtd = sm.graphics.plot_regress_exog(res_glm_PDIxDTD, 'PDI', fig=figsz)

print(res_glm_PDIxDTD.summary())
res_glm_PDIxDTD = pd.read_html(res_glm_PDIxDTD.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxDTD =res_glm_PDIxDTD.assign(y= "DTD").loc[['PDI']]





# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxDTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxDTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxDTD=pd.concat([lb,hb])

#%%

# overadjustment
glm_PDIxOA= sm.GLM(data_glm["ADJ"], pdi_c)
res_glm_PDIxOA = glm_PDIxOA.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxOA.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxOA.summary())
res_glm_PDIxOA = pd.read_html(res_glm_PDIxOA.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxOA =res_glm_PDIxOA.assign(y= "ADJ").loc[['PDI']]

# beta=res_glm_PDIxOA[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxOA[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxOA[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxOA=pd.concat([lb,hb])
#%%
#Prior Confidence
glm_PDIxPrConf = sm.GLM(data_glm["Conf-0"], pdi_c)
res_glm_PDIxPrConf = glm_PDIxPrConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxPrConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

#NON-NORMAL VARIANCE
print(res_glm_PDIxPrConf.summary())
res_glm_PDIxPrConf = pd.read_html(res_glm_PDIxPrConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxPrConf =res_glm_PDIxPrConf.assign(y= "Conf-0").loc[['PDI']]

# beta=res_glm_PDIxPrConf[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxPrConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxPrConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxPrConf=pd.concat([lb,hb])

#%%
# 1ST DRAW Confidence
glm_PDIx1srConf= sm.GLM(data_glm["Conf-1"], pdi_c)
res_glm_PDIx1srConf = glm_PDIx1srConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIx1srConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIx1srConf.summary())
res_glm_PDIx1srConf = pd.read_html(res_glm_PDIx1srConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIx1srConf =res_glm_PDIx1srConf.assign(y= "Conf-1").loc[['PDI']]

# beta=res_glm_PDIx1srConf[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIx1srConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIx1srConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIx1srConf=pd.concat([lb,hb])
#%%
# Final Confidence
glm_PDIxFinConf = sm.GLM(data_glm["Conf-N"], pdi_c)
res_glm_PDIxFinConf= glm_PDIxFinConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxFinConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxFinConf.summary())
res_glm_PDIxFinConf = pd.read_html(res_glm_PDIxFinConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxFinConf =res_glm_PDIxFinConf.assign(y= "Conf-N").loc[['PDI']]

# beta=res_glm_PDIxFinConf[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxFinConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxFinConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxFinConf=pd.concat([lb,hb])

#%%

# Mean RT
glm_PDIxRT= sm.GLM(data_glm["RT"], pdi_c)
res_glm_PDIxRT= glm_PDIxRT.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxRT.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxRT.summary())
res_glm_PDIxRT = pd.read_html(res_glm_PDIxRT.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxRT =res_glm_PDIxRT.assign(y= "RT").loc[['PDI']]

# beta=res_glm_PDIxRT[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxRT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxRT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxRT=pd.concat([lb,hb])

# fig, ax = plt.subplots()

# ax.set_title('Model Fit Plot')
# ax.set_ylabel('Observed values')
# ax.set_xlabel('Fitted values');

# residuals = res_glm_PDIxRT.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()
#%%
# EC Mean RT
glm_PDIxRT_EC= sm.GLM(data_glm["RT_HC"], pdi_c)
res_glm_PDIxRT_EC= glm_PDIxRT_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxRT_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

#ALMOST NORMAL VARIANCE 
print(res_glm_PDIxRT_EC.summary())
res_glm_PDIxRT_EC = pd.read_html(res_glm_PDIxRT_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxRT_EC =res_glm_PDIxRT_EC.assign(y= "RT_HC").loc[['PDI']]

# beta=res_glm_PDIxRT_EC[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxRT_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxRT_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxRT_EC=pd.concat([lb,hb])

#%%
# HC Mean RT
glm_PDIxRT_HC= sm.GLM(data_glm["RT_LC"], pdi_c)
res_glm_PDIxRT_HC= glm_PDIxRT_HC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxRT_HC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxRT_HC.summary())
res_glm_PDIxRT_HC = pd.read_html(res_glm_PDIxRT_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxRT_HC =res_glm_PDIxRT_HC.assign(y= "RT_LC").loc[['PDI']]

# beta=res_glm_PDIxRT_HC[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxRT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxRT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxRT_HC=pd.concat([lb,hb])
#%%

# Mean Accuracy
glm_PDIxAcc = sm.GLM(data_glm["Accuracy"], pdi_c)
res_glm_PDIxAcc= glm_PDIxAcc.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxAcc.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxAcc.summary())
res_glm_PDIxAcc = pd.read_html(res_glm_PDIxAcc.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxAcc =res_glm_PDIxAcc.assign(y= "Accuracy").loc[['PDI']]

# fig, ax = plt.subplots()

# residuals = res_glm_PDIxAcc.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()

# beta=res_glm_PDIxAcc[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxAcc[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxAcc[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxAcc=pd.concat([lb,hb])

#%%
# Mean Accuracy
glm_PDIxAcc_EC = sm.GLM(data_glm["Accuracy_HC"], pdi_c)
res_glm_PDIxAcc_EC= glm_PDIxAcc_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxAcc_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

#NON-NORMAL VARIANCE
print(res_glm_PDIxAcc_EC.summary())
res_glm_PDIxAcc_EC = pd.read_html(res_glm_PDIxAcc_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxAcc_EC =res_glm_PDIxAcc_EC.assign(y= "Accuracy_HC").loc[['PDI']]

# beta=res_glm_PDIxAcc_EC[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxAcc_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxAcc_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxAcc_EC=pd.concat([lb,hb])
#%%
# Mean 
glm_PDIxAcc_HC= sm.GLM(data_glm["Accuracy_LC"], pdi_c)
res_glm_PDIxAcc_HC= glm_PDIxAcc_HC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_PDIxAcc_HC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_PDIxAcc_HC.summary())
res_glm_PDIxAcc_HC = pd.read_html(res_glm_PDIxAcc_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_PDIxAcc_HC =res_glm_PDIxAcc_HC.assign(y= "Accuracy_LC").loc[['PDI']]

# beta=res_glm_PDIxAcc_HC[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_PDIxAcc_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_PDIxAcc_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_PDIxAcc_HC=pd.concat([lb,hb])


res_pdi=pd.concat([res_glm_PDIxRT,res_glm_PDIxRT_EC,res_glm_PDIxRT_HC,
                   res_glm_PDIxAcc,res_glm_PDIxAcc_EC,res_glm_PDIxAcc_HC,
                   res_glm_PDIxDTD, 
                   res_glm_PDIxPrConf,res_glm_PDIx1srConf, res_glm_PDIxFinConf,
                   res_glm_PDIxOA])


res_pdi=res_pdi.assign( Schizotipy = "PDI")
#%%
### GLM CAPS 
caps_c= sm.add_constant(data_glm["CAPS"], prepend=False)

# DTD
glm_CAPSxDTD = sm.GLM(data_glm["DTD"], caps_c)
res_glm_CAPSxDTD = glm_CAPSxDTD.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxDTD.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxDTD.summary())
res_glm_CAPSxDTD = pd.read_html(res_glm_CAPSxDTD.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxDTD =res_glm_CAPSxDTD.assign(y= "DTD").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxDTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxDTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxDTD=pd.concat([lb,hb])
#%%
# overadjustment
glm_CAPSxOA= sm.GLM(data_glm["ADJ"], caps_c)
res_glm_CAPSxOA = glm_CAPSxOA.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxOA.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxOA.summary())
res_glm_CAPSxOA = pd.read_html(res_glm_CAPSxOA.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxOA =res_glm_CAPSxOA.assign(y= "ADJ").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxOA[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxOA[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxOA=pd.concat([lb,hb])
#%%

#Prior Confidence
glm_CAPSxPrConf = sm.GLM(data_glm["Conf-0"], caps_c)
res_glm_CAPSxPrConf = glm_CAPSxPrConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxPrConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxPrConf.summary())
res_glm_CAPSxPrConf = pd.read_html(res_glm_CAPSxPrConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxPrConf =res_glm_CAPSxPrConf.assign(y= "Conf-0").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxPrConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxPrConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxPrConf=pd.concat([lb,hb])
#%%
# 1ST DRAW Confidence
glm_CAPSx1srConf= sm.GLM(data_glm["Conf-1"], caps_c)
res_glm_CAPSx1srConf = glm_CAPSx1srConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSx1srConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSx1srConf.summary())
res_glm_CAPSx1srConf = pd.read_html(res_glm_CAPSx1srConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSx1srConf =res_glm_CAPSx1srConf.assign(y= "Conf-1").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSx1srConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSx1srConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSx1srConf=pd.concat([lb,hb])
#%%
# Final Confidence
glm_CAPSxFinConf = sm.GLM(data_glm["Conf-N"], caps_c)
res_glm_CAPSxFinConf= glm_CAPSxFinConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxFinConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxFinConf.summary())
res_glm_CAPSxFinConf = pd.read_html(res_glm_CAPSxFinConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxFinConf =res_glm_CAPSxFinConf.assign(y= "Conf-N").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxFinConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxFinConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxFinConf=pd.concat([lb,hb])
#%%
# Mean RT
glm_CAPSxRT= sm.GLM(data_glm["RT"], caps_c)
res_glm_CAPSxRT= glm_CAPSxRT.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxRT.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxRT.summary())
res_glm_CAPSxRT = pd.read_html(res_glm_CAPSxRT.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxRT =res_glm_CAPSxRT.assign(y= "RT").loc[['CAPS']]

# fig, ax = plt.subplots()

# residuals = res_glm_CAPSxRT.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxRT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxRT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxRT=pd.concat([lb,hb])
#%%
# Mean RT
glm_CAPSxRT_EC= sm.GLM(data_glm["RT_HC"], caps_c)
res_glm_CAPSxRT_EC= glm_CAPSxRT_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxRT_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxRT_EC.summary())
res_glm_CAPSxRT_EC = pd.read_html(res_glm_CAPSxRT_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxRT_EC =res_glm_CAPSxRT_EC.assign(y= "RT_HC").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxRT_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxRT_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxRT_EC=pd.concat([lb,hb])
#%%
# Mean RT
glm_CAPSxRT_HC= sm.GLM(data_glm["RT_LC"], caps_c)
res_glm_CAPSxRT_HC= glm_CAPSxRT_HC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxRT_HC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxRT_HC.summary())
res_glm_CAPSxRT_HC = pd.read_html(res_glm_CAPSxRT_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxRT_HC =res_glm_CAPSxRT_HC.assign(y= "RT_LC").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxRT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxRT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxRT_HC=pd.concat([lb,hb])
#%%
# Mean Accuracy
glm_CAPSxAcc = sm.GLM(data_glm["Accuracy"], caps_c)
res_glm_CAPSxAcc= glm_CAPSxAcc.fit()
#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxAcc.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxAcc.summary())
res_glm_CAPSxAcc = pd.read_html(res_glm_CAPSxAcc.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxAcc =res_glm_CAPSxAcc.assign(y= "Accuracy").loc[['CAPS']]


# residuals = res_glm_CAPSxAcc.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()

# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxAcc[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxAcc[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxAcc=pd.concat([lb,hb])
#%%
# Mean Accuracy
glm_CAPSxAcc_EC = sm.GLM(data_glm["Accuracy_HC"], caps_c)
res_glm_CAPSxAcc_EC= glm_CAPSxAcc_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxAcc_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxAcc_EC.summary())
res_glm_CAPSxAcc_EC = pd.read_html(res_glm_CAPSxAcc_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxAcc_EC =res_glm_CAPSxAcc_EC.assign(y= "Accuracy_HC").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxAcc_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxAcc_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxAcc_EC=pd.concat([lb,hb])
#%%
# Mean Accuracy
glm_CAPSxAcc_HC = sm.GLM(data_glm["Accuracy_LC"], caps_c)
res_glm_CAPSxAcc_HC= glm_CAPSxAcc_HC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_CAPSxAcc_HC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_CAPSxAcc_HC.summary())
res_glm_CAPSxAcc_HC = pd.read_html(res_glm_CAPSxAcc_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_CAPSxAcc_HC =res_glm_CAPSxAcc_HC.assign(y= "Accuracy_LC").loc[['CAPS']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_CAPSxAcc_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_CAPSxAcc_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_CAPSxAcc_HC=pd.concat([lb,hb])


res_caps=pd.concat([res_glm_CAPSxRT,res_glm_CAPSxRT_EC,res_glm_CAPSxRT_HC,
                   res_glm_CAPSxAcc,res_glm_CAPSxAcc_EC,res_glm_CAPSxAcc_HC,
                   res_glm_CAPSxDTD, 
                   res_glm_CAPSxPrConf,res_glm_CAPSx1srConf, res_glm_CAPSxFinConf,
                   res_glm_CAPSxOA])

res_caps=res_caps.assign( Schizotipy = "CAPS")
#%%%
### GLM ASI 
asi_c= sm.add_constant(data_glm["ASI"], prepend=False)

# DTD
glm_ASIxDTD = sm.GLM(data_glm["DTD"], asi_c)
res_glm_ASIxDTD = glm_ASIxDTD.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxDTD.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxDTD.summary())
res_glm_ASIxDTD = pd.read_html(res_glm_ASIxDTD.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxDTD =res_glm_ASIxDTD.assign(y= "DTD").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxDTD[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxDTD[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxDTD=pd.concat([lb,hb])
#%%
# overadjustment
glm_ASIxOA= sm.GLM(data_glm["ADJ"], asi_c)
res_glm_ASIxOA = glm_ASIxOA.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxOA.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxOA.summary())
res_glm_ASIxOA = pd.read_html(res_glm_ASIxOA.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxOA =res_glm_ASIxOA.assign(y= "ADJ").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxOA[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxOA[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxOA=pd.concat([lb,hb])
#%%
#Prior Confidence
glm_ASIxPrConf = sm.GLM(data_glm["Conf-0"], asi_c)
res_glm_ASIxPrConf = glm_ASIxPrConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxPrConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxPrConf.summary())
res_glm_ASIxPrConf = pd.read_html(res_glm_ASIxPrConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxPrConf =res_glm_ASIxPrConf.assign(y= "Conf-0").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxPrConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxPrConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxPrConf=pd.concat([lb,hb])
#%%
# 1ST DRAW Confidence
glm_ASIx1srConf= sm.GLM(data_glm["Conf-1"], asi_c)
res_glm_ASIx1srConf = glm_ASIx1srConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIx1srConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIx1srConf.summary())
res_glm_ASIx1srConf = pd.read_html(res_glm_ASIx1srConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIx1srConf =res_glm_ASIx1srConf.assign(y= "Conf-1").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIx1srConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIx1srConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIx1srConf=pd.concat([lb,hb])
#%%
# Final Confidence
glm_ASIxFinConf = sm.GLM(data_glm["Conf-N"], asi_c)
res_glm_ASIxFinConf= glm_ASIxFinConf.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxFinConf.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxFinConf.summary())
res_glm_ASIxFinConf = pd.read_html(res_glm_ASIxFinConf.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxFinConf =res_glm_ASIxFinConf.assign(y= "Conf-N").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxFinConf[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxFinConf[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxFinConf=pd.concat([lb,hb])
#%%
# Mean RT
glm_ASIxRT= sm.GLM(data_glm["RT"], asi_c)
res_glm_ASIxRT= glm_ASIxRT.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxRT.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxRT.summary())

# fig, ax = plt.subplots()

# residuals = res_glm_ASIxRT.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()

res_glm_ASIxRT = pd.read_html(res_glm_ASIxRT.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxRT =res_glm_ASIxRT.assign(y= "RT").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxRT[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxRT[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxRT=pd.concat([lb,hb])
#%%

# Mean RT
glm_ASIxRT_EC= sm.GLM(data_glm["RT_HC"], asi_c)
res_glm_ASIxRT_EC= glm_ASIxRT_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxRT_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxRT_EC.summary())
res_glm_ASIxRT_EC = pd.read_html(res_glm_ASIxRT_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxRT_EC =res_glm_ASIxRT_EC.assign(y= "RT_HC").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxRT_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxRT_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxRT_EC=pd.concat([lb,hb])
#%%
# Mean RT
glm_ASIxRT_HC= sm.GLM(data_glm["RT_LC"], asi_c)
res_glm_ASIxRT_HC= glm_ASIxRT_HC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxRT_HC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxRT_HC.summary())
res_glm_ASIxRT_HC = pd.read_html(res_glm_ASIxRT_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxRT_HC =res_glm_ASIxRT_HC.assign(y= "RT_LC").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxRT_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxRT_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxRT_HC=pd.concat([lb,hb])
#%%
# Mean Accuracy
glm_ASIxAcc= sm.GLM(data_glm["Accuracy"], asi_c)
res_glm_ASIxAcc= glm_ASIxAcc.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxAcc.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxAcc.summary())
res_glm_ASIxAcc = pd.read_html(res_glm_ASIxAcc.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxAcc =res_glm_ASIxAcc.assign(y= "Accuracy").loc[['ASI']]
# fig, ax = plt.subplots()

# residuals = res_glm_ASIxAcc.resid_response
# plt.scatter(pdi_c.PDI, residuals)
# plt.title('Residual Plot')
# plt.xlabel('x')
# plt.ylabel('Residuals')
# plt.show()

# res_glm_ASIxAcc = pd.read_html(res_glm_ASIxAcc.summary().tables[1].as_html(),header=0,index_col=0)[0]
# res_glm_ASIxAcc =res_glm_ASIxAcc.assign(y= "Accuracy").loc[['ASI']]


# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxAcc[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxAcc[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxAcc=pd.concat([lb,hb])
#%%
# Mean Accuracy
glm_ASIxAcc_EC = sm.GLM(data_glm["Accuracy_HC"], asi_c)
res_glm_ASIxAcc_EC= glm_ASIxAcc_EC.fit()

#plot residuals
fig, ax = plt.subplots()
resid = res_glm_ASIxAcc_EC.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax.hist(resid_std, bins=25)
ax.set_title('Histogram of standardized deviance residuals');
graphics.gofplots.qqplot(resid, line='r')

print(res_glm_ASIxAcc_EC.summary())
res_glm_ASIxAcc_EC = pd.read_html(res_glm_ASIxAcc_EC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxAcc_EC =res_glm_ASIxAcc_EC.assign(y= "Accuracy_HC").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxAcc_EC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxAcc_EC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxAcc_EC=pd.concat([lb,hb])

# Mean Accuracy
glm_ASIxAcc_HC = sm.GLM(data_glm["Accuracy_LC"], asi_c)
res_glm_ASIxAcc_HC= glm_ASIxAcc_HC.fit()
print(res_glm_ASIxAcc_HC.summary())
res_glm_ASIxAcc_HC = pd.read_html(res_glm_ASIxAcc_HC.summary().tables[1].as_html(),header=0,index_col=0)[0]
res_glm_ASIxAcc_HC =res_glm_ASIxAcc_HC.assign(y= "Accuracy_LC").loc[['ASI']]
# beta=res_glm_PDIxDTD[col]
# beta.rename(columns = {'coef':'b'}, inplace = True)
lb=res_glm_ASIxAcc_HC[col1]
lb.rename(columns = {'[0.025':'b'}, inplace = True)
hb=res_glm_ASIxAcc_HC[col2]
hb.rename(columns = {'0.975]':'b'}, inplace = True)
res_glm_ASIxAcc_HC=pd.concat([lb,hb])


res_asi=pd.concat([res_glm_ASIxRT,
                   res_glm_ASIxRT_EC,
                   res_glm_ASIxRT_HC,
                   res_glm_ASIxAcc,
                   res_glm_ASIxAcc_EC,
                   res_glm_ASIxAcc_HC,
                   res_glm_ASIxDTD, 
                   res_glm_ASIxPrConf,
                   res_glm_ASIx1srConf, 
                   res_glm_ASIxFinConf,
                   res_glm_ASIxOA])

res_asi=res_asi.assign( Schizotipy = "ASI")
#%%
res_comp=pd.concat([res_pdi, res_caps, res_asi])



res_comp=res_comp.replace({'RT':'RT µ',
                         'RT_HC':'HC RT µ',
                         'RT_LC':'LC RT µ',
                         'Accuracy':'Accuracy µ',
                         'Accuracy_HC':'HC Accuracy µ',
                         'Accuracy_LC':'LC Accuracy µ',
                         'Conf-0':'Conf-0',
                         'Conf-1':'Conf-1',
                         'Conf-N':'Conf-N',
                         'ADJ':'ADJ'})

res_comp.to_csv('study1_results_glm_2024.csv')
#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# res_comp=pd.read_csv('study1_results_glm.csv')
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
                         'b':'β'},
                 width=1500,
                height=1000)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


# iterate on each region
for i in res_comp["y"].unique():
    # filter by region
    df_sub = res_comp[res_comp["y"] == i]
    
    fig.add_shape(
        type="line",
        layer="below",
         xref='x1',line=dict(color=coloursdist['PDI'],width=3.5),
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
         xref='x2',line=dict(color=coloursdist['CAPS'],width=3.5),
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
         xref='x3',line=dict(color=coloursdist['ASI'],width=3.5),
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
      text='<b>*<b>',
      showarrow=False,
      font=dict(size=22, color="black", family="Courier New, monospace"),
      xref='x2',)    

fig.add_annotation(
      y='Conf-N', x=0.5,
      text='<b>**<b>',
      showarrow=False,
      font=dict(size=22, color="black", family="Courier New, monospace"),
      xref='x3',)    


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=35,
    title_font_family="Times New Roman",
    title_font_color="black",
   legend_title_font_color="blue",
    legend_font_size=35,
    width=1600,  # Set your desired width in pixels
    height=1300  # Set your desired height in pixels
)

fig.show()

fig.write_image("study1_fig_glm_A_2024.png",scale=3)

















