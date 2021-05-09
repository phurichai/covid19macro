# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 23:18:37 2021

@author: Phurichai
"""
import time
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from datetime import date, datetime, timedelta
import country_converter as coco 
import seaborn as sns


tic = time.perf_counter()

# ----------------------------------------                                                                             
#   Full data update 
# ----------------------------------------
# !!!! Specify 'chosen_date' in param_simple before running !!!!
import data_covid                                                                                   
from param_simple import *
from seir_simple import *

# ---------------------------------------------
#       Functions to generate figures
# ---------------------------------------------
sns.set_theme()
def fig2_baseline(cset=['US','DE','GB','FR']):
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()
        name = f'../output/{out_save_folder}/{c}_baseline.pkl'
        pickle.dump(tmp,open(name,'wb'))
        fig = tmp.plot_portrait(saveplot=False)
        Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
        fig.savefig(f'../pics/paper_{date.today()}/fig2-baseline-{c}.pdf')  
        
def fig_update_multi(cset=['US','DE','GB','FR'],fname='multi'):
    transpa = 0.0
    color2 = 'dodgerblue'
    cvec = ['firebrick', 'darkorange', 'cornflowerblue', 'seagreen']
    cn = 0
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), constrained_layout=True)
    for c in cset:
        # tmp = solveCovid(c)
        # tmp.prelim()
        # tmp.gamma_t_compute()
        # tmp.fitmodel()
        # tmp.sim_seir()
        tmp = pickle.load(open(f'../output/{out_load_folder}/{c}_baseline.pkl','rb'))
        df = tmp.df3
        ax[0,0].plot(df.index, 100*df['I']/tmp.N, label=c, color=cvec[cn])
        ax[0,1].plot(df.index, 100*df['mob_fc'], label=c, color=cvec[cn])
        ax[1,0].plot(df.index, 100*df['V']/tmp.N, label=c, color=cvec[cn])
        ax[1,1].plot(df.index, 100*df['S']/tmp.N, label=c, color=cvec[cn])
        cn +=1 
    ax[0,0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[0,0].set_title('Infectious population',fontsize='x-large')
    ax[0,0].set(ylabel = '% of population')    
    ax[0,0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[0,1].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[0,1].set_title('Mobility',fontsize='x-large')
    ax[0,1].set(ylabel = '% deviations from norm') 
    ax[0,1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[1,0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[1,0].set_title('Vaccinated population',fontsize='x-large')
    ax[1,0].set(ylabel = '% of population') 
    ax[1,0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[1,1].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[1,1].set_title('Susceptible population',fontsize='x-large')
    ax[1,1].set(ylabel = '% of population') 
    ax[1,1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    plt.setp(ax[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    fn_text = f'Estimate as of {date.today()}'
    plt.figtext(0.2, -0.05, fn_text, horizontalalignment='center') 
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig-update-{fname}-{date.today()}.png') 
  

# ===========================
#       Calling codes 
# ===========================

cset = ['AR','AU','BE','BR','CA','FR','DE',      
          'IN','ID','IT','JP','KR','MY','MX',
          'NL','PL','RU','SA','SG','ZA','ES','SE',
          'CH','TH','TR','GB','US']
fig2_baseline(cset)

fig_update_multi(['US','DE','GB','FR'],'ADV')
fig_update_multi(['BR','IN','KR','ZA'],'EME')

cset2 = ['US','DE','GB','FR','ES','IT','CH','JP',
         'BR','MX','IN','KR','ZA']

df_update = update_table(cset2)
print(df_update.to_markdown(tablefmt="grid"))

# Save estimated parameters and simulation results in spreadsheets 
# run_baseline(cset)
# save_results(cset)