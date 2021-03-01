# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:12:32 2020

@author: Phurichai Rungcharoenkitkul

Compute adjustment factor for targeted vaccines by countries
Goal: 
    - Map average p_dth to int(p_d,age * F(age))
    - F(age) is from curve-fitting, using demographic data
    - p_d,age is from study Megan et al
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import country_converter as coco 
from country import country
import pdb
import pickle

#--------------------
# Define country list
c_set = country('special1','UNcode')

age_0 = np.array([7, 85])
fatality = np.array([0.001, 8.29])/100 # From https://doi.org/10.1038/s41586-020-2918-0

log_fatality = np.log(fatality)
# Calculate p_d,age global; IFR is log-linear in age buckets
fat_func = interp1d(age_0,log_fatality,fill_value="extrapolate")
age_new = np.linspace(0, 100, 500)
fat_new = np.exp(fat_func(age_new))

#plt.plot(age_new, fat_new)

# Demographic data
df = pd.read_excel("../data/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx",sheet_name="ESTIMATES",
                    skiprows=range(1,16),usecols='C,E,H:AC',header=1, index_col=0,
                    engine='openpyxl',)
df1 = df[(df['Reference date (as of 1 July)']==2020) & (df['Country code'].isin(c_set))]
df1['iso2'] = coco.convert(names=df1['Country code'].values.tolist(), to='ISO2')
df1 = df1.set_index('iso2')
df1 = df1.drop(columns=['Country code', 'Reference date (as of 1 July)'])

mid = [2,7,12,17,22,27,32,37,42,47,52,57,62,67,72,77,82,87,92,97,102]
demo_interp = {}
table_out = {}
c_set2 = df1.index.tolist()
for c in c_set2:
    val0 = df1.loc[df1.index == c, :].values[0]
    val = val0/val0.sum()
    demo_func = interp1d(mid,val,fill_value='extrapolate')
    demo_interp[c] = demo_func
    # Table of output: outcomes when you save people age y and above
    density_interp = demo_func(age_new)/demo_func(age_new).sum()
    fatality_base = (density_interp*fat_new).sum()
    df2 = pd.DataFrame(columns =['age_cutoff','vaccine_%','fatality_ratio'])
    for age in age_new: # Suppose we save people older than 'age'
        idx = (np.abs(age_new - age)).argmin()
        vac_percent = density_interp[idx:].sum()
        fatality_new = (density_interp[:idx]*fat_new[:idx]).sum()
        fatality_ratio = fatality_new/fatality_base
        df2 = df2.append({'age_cutoff' : age, 'vaccine_%' : vac_percent, 'fatality_ratio' : fatality_ratio},  
                ignore_index = True) 
    table_out[c] = df2

c='US'
plt.plot(table_out[c]['vaccine_%'], table_out[c]['fatality_ratio'])
plt.show()

name = '../data/age_fatality.pkl'
pickle.dump(table_out,open(name,'wb'))



