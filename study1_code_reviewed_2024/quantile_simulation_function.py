#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:56:10 2023

@author: francescoscaramozzino
"""




import pandas as pd
import numpy as np
#####################
#potting with plotly
import plotly.io as pio
pio.renderers.default='browser'


def get_mean_sim_condition_quantiles(simulated_data, condition1, condition2):
    for i in simulated_data:
        levels_condition1=i[condition1].unique()
        levels_condition2=i[condition2].unique()
        
        level_a_condition1=levels_condition1[0]
        level_b_condition1=levels_condition1[1]
        level_a_condition2=levels_condition2[0]
        level_b_condition2=levels_condition2[1]

        data_sim_err = i[i['response'] == 0]
        data_sim_cor = i[i['response'] == 1] 
        
        data_err_high_pdi = data_sim_err[data_sim_err[condition1] == level_a_condition1]
        data_err_high_pdi_15= data_err_high_pdi[data_err_high_pdi[condition2] == level_a_condition2]
        data_err_high_pdi_25= data_err_high_pdi[data_err_high_pdi[condition2] == level_b_condition2]
                      
        data_cor_high_pdi = data_sim_cor[data_sim_cor[condition1] == level_a_condition1]
        data_cor_high_pdi_15= data_cor_high_pdi[data_cor_high_pdi[condition2] == level_a_condition2]
        data_cor_high_pdi_25= data_cor_high_pdi[data_cor_high_pdi[condition2] == level_b_condition2]
        
        data_err_low_pdi = data_sim_err[data_sim_err[condition1] == level_b_condition1]
        data_err_low_pdi_15= data_err_low_pdi[data_err_low_pdi[condition2] == level_a_condition2]
        data_err_low_pdi_25= data_err_low_pdi[data_err_low_pdi[condition2] == level_b_condition2]
        
        data_cor_low_pdi = data_sim_cor[data_sim_cor[condition1] == level_b_condition1]
        data_cor_low_pdi_15= data_cor_low_pdi[data_cor_low_pdi[condition2] == level_a_condition2]
        data_cor_low_pdi_25= data_cor_low_pdi[data_cor_low_pdi[condition2] == level_b_condition2]
                
        sim_ab_0 = np.quantile(data_err_high_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_ab_1 = np.quantile(data_cor_high_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_aa_0 = np.quantile(data_err_low_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_aa_1= np.quantile(data_cor_low_pdi_15['rt'], np.linspace(0, 1, num=11))
        
        sim_bb_0 = np.quantile(data_err_high_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_bb_1 = np.quantile(data_cor_high_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_ba_0 = np.quantile(data_err_low_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_ba_1 = np.quantile(data_cor_low_pdi_25['rt'], np.linspace(0, 1, num=11))
        
        zipped=list(zip(sim_ab_0,sim_ab_1,sim_aa_0,sim_aa_1,
                        sim_bb_0,sim_bb_1,sim_ba_0,sim_ba_1))
        quantiles=pd.DataFrame(zipped, columns=['sim_ab_0','sim_ab_1',
                                         'sim_aa_0','sim_aa_1',
                       'sim_bb_0','sim_bb_1','sim_ba_0','sim_ba_1'])
        
        aa_0='_0_'.join([level_a_condition1,level_a_condition2])
        ab_0='_0_'.join([level_a_condition1,level_b_condition2])
        ba_0='_0_'.join([level_b_condition1,level_a_condition2])
        bb_0='_0_'.join([level_b_condition1,level_b_condition2])
        
        aa_1='_1_'.join([level_a_condition1,level_a_condition2])
        ab_1='_1_'.join([level_a_condition1,level_b_condition2])
        ba_1='_1_'.join([level_b_condition1,level_a_condition2])
        bb_1='_1_'.join([level_b_condition1,level_b_condition2])
        
        quantiles.rename(columns = {'sim_ab_0':ab_0,'sim_ab_1':ab_1,'sim_aa_0':aa_0,'sim_aa_1':aa_1,
                       'sim_bb_0':bb_0,'sim_bb_1':bb_1,'sim_ba_0':ba_0,'sim_ba_1':ba_1}, inplace = True)

        
        quantile = np.array(['0.5_q','10_q','20_q','30_q','40_q','50_q','60_q','70_q','80_q','90_q','99_q'])
       
        
        quantiles['quantile'] = quantile.tolist()
       
        simulated_quantiles = []
        simulated_quantiles.append(quantiles)
    mean_simulated_quantiles = pd.concat(simulated_quantiles).groupby('quantile').mean()
    mean_simulated_quantiles['quantile'] = quantile.tolist()
    return mean_simulated_quantiles


def get_mean_sim_1condition_quantiles(simulated_data, condition1):
    for i in simulated_data:
        levels_condition1=i[condition1].unique()
        
        level_a_condition1=levels_condition1[0]
        level_b_condition1=levels_condition1[1]

        data_sim_err = i[i['response'] == 0]
        data_sim_cor = i[i['response'] == 1] 
        
        data_err_high_pdi = data_sim_err[data_sim_err[condition1] == level_a_condition1]
                      
        data_cor_high_pdi = data_sim_cor[data_sim_cor[condition1] == level_a_condition1]
        
        data_err_low_pdi = data_sim_err[data_sim_err[condition1] == level_b_condition1]
        
        data_cor_low_pdi = data_sim_cor[data_sim_cor[condition1] == level_b_condition1]
                
        sim_a_0 = np.quantile(data_err_high_pdi['rt'], np.linspace(0, 1, num=11))
        sim_a_1 = np.quantile(data_cor_high_pdi['rt'], np.linspace(0, 1, num=11))
        sim_b_0 = np.quantile(data_err_low_pdi['rt'], np.linspace(0, 1, num=11))
        sim_b_1= np.quantile(data_cor_low_pdi['rt'], np.linspace(0, 1, num=11))
        
        zipped=list(zip(sim_a_0,sim_a_1,sim_b_0,sim_b_1))
        quantiles=pd.DataFrame(zipped, columns=['sim_a_1','sim_a_0',
                                         'sim_b_1','sim_b_0'])
        
        a_0=''.join([level_a_condition1,'_0'])
        b_0=''.join([level_b_condition1,'_0'])
        
        a_1=''.join([level_a_condition1,'_1'])
        b_1=''.join([level_b_condition1,'_1'])
        
        quantiles.rename(columns = {'sim_a_1':a_1,'sim_a_0':a_0,'sim_b_1':b_1,'sim_b_0':b_0}, inplace = True)

        
        quantile = np.array(['0.5_q','10_q','20_q','30_q','40_q','50_q','60_q','70_q','80_q','90_q','99_q'])
       
        
        quantiles['quantile'] = quantile.tolist()
       
        simulated_quantiles = []
        simulated_quantiles.append(quantiles)
    mean_simulated_quantiles = pd.concat(simulated_quantiles).groupby('quantile').mean()
    mean_simulated_quantiles['quantile'] = quantile.tolist()
    return mean_simulated_quantiles


def get_mean_sim_quantiles(simulated_data):
    for i in simulated_data:
       
        data_sim_err = i[i['response'] == 0]
        data_sim_cor = i[i['response'] == 1] 
        
        q_data_sim_cor = np.quantile(data_sim_cor['rt'], np.linspace(0.10, 0.90, num=5))
        q_data_sim_err = np.quantile(data_sim_err['rt'], np.linspace(0.10, 0.90, num=5))
        
        zipped=list(zip(q_data_sim_cor,q_data_sim_err))
        quantiles=pd.DataFrame(zipped, columns=['q_data_sim_cor','q_data_sim_err'])
        
        quantiles.rename(columns = {'q_data_sim_cor':'q_correct',
                                    'q_data_sim_err':'q_incorrect'}, inplace = True)

        
        quantile = np.array(['10_q','30_q','50_q','70_q','90_q'])
        
        quantiles['quantile'] = quantile.tolist()
       
        simulated_quantiles = []
        simulated_quantiles.append(quantiles)
    mean_simulated_quantiles = pd.concat(simulated_quantiles).groupby('quantile').mean()
    mean_simulated_quantiles['quantile'] = quantile.tolist()
    return mean_simulated_quantiles


def get_condition_quantiles(data, condition1, condition2):
        levels_condition1=data[condition1].unique()
        levels_condition2=data[condition2].unique()
        
        level_a_condition1=levels_condition1[0]
        level_b_condition1=levels_condition1[1]
        level_a_condition2=levels_condition2[0]
        level_b_condition2=levels_condition2[1]

        data_sim_err = data[data['response'] == 0]
        data_sim_cor = data[data['response'] == 1] 
        
        data_err_high_pdi = data_sim_err[data_sim_err[condition1] == level_a_condition1]
        data_err_high_pdi_15= data_err_high_pdi[data_err_high_pdi[condition2] == level_a_condition2]
        data_err_high_pdi_25= data_err_high_pdi[data_err_high_pdi[condition2] == level_b_condition2]
                      
        data_cor_high_pdi = data_sim_cor[data_sim_cor[condition1] == level_a_condition1]
        data_cor_high_pdi_15= data_cor_high_pdi[data_cor_high_pdi[condition2] == level_a_condition2]
        data_cor_high_pdi_25= data_cor_high_pdi[data_cor_high_pdi[condition2] == level_b_condition2]
        
        data_err_low_pdi = data_sim_err[data_sim_err[condition1] == level_b_condition1]
        data_err_low_pdi_15= data_err_low_pdi[data_err_low_pdi[condition2] == level_a_condition2]
        data_err_low_pdi_25= data_err_low_pdi[data_err_low_pdi[condition2] == level_b_condition2]
        
        data_cor_low_pdi = data_sim_cor[data_sim_cor[condition1] == level_b_condition1]
        data_cor_low_pdi_15= data_cor_low_pdi[data_cor_low_pdi[condition2] == level_a_condition2]
        data_cor_low_pdi_25= data_cor_low_pdi[data_cor_low_pdi[condition2] == level_b_condition2]
                
        sim_ab_0 = np.quantile(data_err_high_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_ab_1 = np.quantile(data_cor_high_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_aa_0 = np.quantile(data_err_low_pdi_15['rt'], np.linspace(0, 1, num=11))
        sim_aa_1= np.quantile(data_cor_low_pdi_15['rt'], np.linspace(0, 1, num=11))
        
        sim_bb_0 = np.quantile(data_err_high_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_bb_1 = np.quantile(data_cor_high_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_ba_0 = np.quantile(data_err_low_pdi_25['rt'], np.linspace(0, 1, num=11))
        sim_ba_1 = np.quantile(data_cor_low_pdi_25['rt'], np.linspace(0, 1, num=11))
        
        zipped=list(zip(sim_ab_0,sim_ab_1,sim_aa_0,sim_aa_1,
                        sim_bb_0,sim_bb_1,sim_ba_0,sim_ba_1))
        quantiles=pd.DataFrame(zipped, columns=['sim_ab_0','sim_ab_1',
                                         'sim_aa_0','sim_aa_1',
                       'sim_bb_0','sim_bb_1','sim_ba_0','sim_ba_1'])
        
        aa_0='_0_'.join([level_a_condition1,level_a_condition2])
        ab_0='_0_'.join([level_a_condition1,level_b_condition2])
        ba_0='_0_'.join([level_b_condition1,level_a_condition2])
        bb_0='_0_'.join([level_b_condition1,level_b_condition2])
        
        aa_1='_1_'.join([level_a_condition1,level_a_condition2])
        ab_1='_1_'.join([level_a_condition1,level_b_condition2])
        ba_1='_1_'.join([level_b_condition1,level_a_condition2])
        bb_1='_1_'.join([level_b_condition1,level_b_condition2])
        
        quantiles.rename(columns = {'sim_ab_0':ab_0,'sim_ab_1':ab_1,'sim_aa_0':aa_0,'sim_aa_1':aa_1,
                       'sim_bb_0':bb_0,'sim_bb_1':bb_1,'sim_ba_0':ba_0,'sim_ba_1':ba_1}, inplace = True)

        
        quantile = np.array(['0.5_q','10_q','20_q','30_q','40_q','50_q','60_q','70_q','80_q','90_q','99_q'])
        quantiles['quantile'] = quantile.tolist()
        return quantiles
    


def get_quantiles(data):
        
        data_sim_err = data[data['response'] == 0]
        data_sim_cor = data[data['response'] == 1] 
        
        
        q_data_sim_cor = np.quantile(data_sim_cor['rt'], np.linspace(0.10, 0.90, num=5))
        q_data_sim_err = np.quantile(data_sim_err['rt'], np.linspace(0.10, 0.90, num=5))
        
        zipped=list(zip(q_data_sim_cor,q_data_sim_err))
        quantiles=pd.DataFrame(zipped, columns=['q_data_sim_cor','q_data_sim_err'])
        
        quantiles.rename(columns = {'q_data_sim_cor':'q_correct',
                                    'q_data_sim_err':'q_incorrect'}, inplace = True)

        
        quantile = np.array(['10_q','30_q','50_q','70_q','90_q'])
        quantiles['quantile'] = quantile.tolist()
        return quantiles