# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 08:06:35 2021

@author: Phurichai Rungcharoenkitkul
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

def fig1(cset=['US','DE','GB','KR']):
    scale = 1.2
    cvec = ['firebrick', 'darkorange', 'cornflowerblue', 'seagreen']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(scale*5*2,scale*4), constrained_layout=True)
    transpa = 0.1
    cn = 0
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        df = tmp.df2
        df['gamma_t_sm'] = uniform_filter1d(df['gamma_t'], size=7, mode='nearest')
        ax[0].plot(df.index, df['gamma_t_sm'], label=c,  color=cvec[cn], linestyle='-')
        ax[1].plot(df.index, df['pdth_t'], label=c, color=cvec[cn], linestyle='-')
        cn += 1
    ax[0].set_title('Infection rate, $\gamma_t$',fontsize='x-large')
    ax[0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[1].set_title('Fatality probability, $p_{dth,t}$',fontsize='x-large')
    ax[1].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    # plt.setp(ax[0].get_xticklabels(), rotation=30, horizontalalignment='right')
    # plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig1.pdf')   


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

def fig2_baseline_multi(cset=['US','DE','GB','FR'],fname='multi'):
    transpa = 0.0
    color2 = 'dodgerblue'
    cvec = ['firebrick', 'darkorange', 'cornflowerblue', 'seagreen']
    cn = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout=True)
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()
        df = tmp.df3
        ax[0].plot(df.index, 100*df['I']/tmp.N, label=c, color=cvec[cn])
        ax[1].plot(df.index, 100*df['mob_fc'], label=c, color=cvec[cn])
        cn +=1 
    ax[0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[0].set_title('Infectious group ($I_t/N$)',fontsize='x-large')
    ax[0].set(ylabel = '% of population')    
    ax[0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[1].legend(loc='best',framealpha=transpa ,fontsize='x-large')
    ax[1].set_title('Mobility',fontsize='x-large')
    ax[1].set(ylabel = '% deviations from norm') 
    ax[1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    plt.setp(ax[0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig2-baseline-{fname}.pdf')          
        
def fig3_spike(cset = ['US','DE','GB']):
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()                
        df = tmp.df3
        
        t_spike = solveCovid(c)        
        t_spike.prelim()
        t_spike.gamma_t_compute()
        t_spike.fitmodel()
        t_spike.gamma_tilde_model = 'shock'
        t_spike.sim_seir()                
        dfb = t_spike.df3
        name = f'../output/{out_save_folder}/{c}_shock.pkl'
        pickle.dump(t_spike,open(name,'wb'))
        
        transpa = 0.0
        color2 = 'dodgerblue'
        color3 = 'dimgray'
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout=True)
        
        ax[0].plot(df.index, df['gamma_t_fc'],label=r'$\gamma_t$: baseline', color='red')
        ax[0].plot(dfb.index, dfb['gamma_t_fc'],label=r'$\gamma_t$: spike',color=color2)
        ax[0].plot(df.index, df['gamma_t'], color=color3)
        ax[0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[0].set_title('Infection rate',fontsize='x-large')
        ax[0].legend(loc='best',framealpha=transpa ,fontsize='x-large')        
    
        ax[1].plot(df.index, 100*df['mob_fc'], label='Baseline', color='red')
        ax[1].plot(dfb.index, 100*dfb['mob_fc'], label='Infection spike', color=color2)
        ax[1].plot(df.index, 100*df['google_smooth'], color=color3)
        ax[1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[1].legend(loc='lower right',framealpha=transpa ,fontsize='x-large')
        ax[1].set_title('Mobility',fontsize='x-large')
        ax[1].set(ylabel='% deviations from norm')
        
        cname = coco.convert(names=tmp.iso2,to='name_short')
        fig.suptitle(f'{cname}',fontsize=18) 

        plt.setp(ax[0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')
        
        Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
        fig.savefig(f'../pics/paper_{date.today()}/fig3-spike-{c}.pdf')  


def fig4_vac(cset = ['US','DE','GB']):
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()                
        df = tmp.df3

        t_vac = solveCovid(c)
        t_vac.vac_assump = 'vac_worse'
        t_vac.prelim()
        t_vac.gamma_t_compute()
        t_vac.fitmodel()
        t_vac.sim_seir()
        dfb = t_vac.df3
        name = f'../output/{out_save_folder}/{c}_vacworse.pkl'
        pickle.dump(t_vac,open(name,'wb'))
        
        transpa = 0.0
        color2 = 'dodgerblue'
        color3 = 'dimgray'
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout=True)
        
        ax[0].plot(df.index, 100*df['S']/tmp.N, label='$S_t$',color='red')
        ax[0].plot(df.index, 100*df['V']/tmp.N, label='$V_t$',color='red',linestyle=':')
        ax[0].plot(df.index, 100*dfb['S']/t_vac.N, label='$S_t$: low vaccines',color=color2)
        ax[0].plot(df.index, 100*dfb['V']/t_vac.N, label='$V_t$: low vaccines',color=color2,linestyle=':')
        ax[0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[0].set_title('Susceptible & vaccinated',fontsize='x-large')
        ax[0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        ax[0].set(ylabel='% of population')
    
        ax[1].plot(df.index, 100*df['mob_fc'], label='Baseline', color='red')
        ax[1].plot(dfb.index, 100*dfb['mob_fc'], label='Low vaccines', color=color2)
        ax[1].plot(df.index, 100*df['google_smooth'], color=color3)
        ax[1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[1].legend(loc='lower right',framealpha=transpa ,fontsize='x-large')
        ax[1].set_title('Mobility',fontsize='x-large')
        ax[1].set(ylabel='% deviations from norm')

        cname = coco.convert(names=tmp.iso2,to='name_short')
        fig.suptitle(f'{cname}',fontsize=18) 

        plt.setp(ax[0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')
        
        Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
        fig.savefig(f'../pics/paper_{date.today()}/fig4-vacworse-{c}.pdf')  


def fig5_reinfect(cset = ['US','DE','GB']):
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()                
        df = tmp.df3

        t_reinfect = solveCovid(c)
        t_reinfect.reinfect = 'reinfect'
        t_reinfect.prelim()
        t_reinfect.gamma_t_compute()
        t_reinfect.fitmodel()
        t_reinfect.sim_seir()
        dfb = t_reinfect.df3
        name = f'../output/{out_save_folder}/{c}_reinfect.pkl'
        pickle.dump(t_reinfect,open(name,'wb'))
        
        transpa = 0.0
        color2 = 'dodgerblue'
        color3 = 'dimgray'
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout=True)
        
        ax[0].plot(df.index, 100*df['S']/tmp.N, label='$S_t$',color='red')
        ax[0].plot(df.index, 100*df['V']/tmp.N, label='$V_t$',color='red',linestyle=':')
        ax[0].plot(df.index, 100*dfb['S']/t_reinfect.N, label='$S_t$: reinfect',color=color2)
        ax[0].plot(df.index, 100*dfb['V']/t_reinfect.N, label='$V_t$: reinfect',color=color2,linestyle=':')
        ax[0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[0].set_title('Susceptible & vaccinated',fontsize='x-large')
        ax[0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        ax[0].set(ylabel='% of population')
    
        ax[1].plot(df.index, 100*df['mob_fc'], label='Baseline', color='red')
        ax[1].plot(dfb.index, 100*dfb['mob_fc'], label='Reinfection', color=color2)
        ax[1].plot(df.index, 100*df['google_smooth'], color=color3)
        ax[1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[1].legend(loc='lower right',framealpha=transpa ,fontsize='x-large')
        ax[1].set_title('Mobility',fontsize='x-large')
        ax[1].set(ylabel='% deviations from norm')

        cname = coco.convert(names=tmp.iso2,to='name_short')
        fig.suptitle(f'{cname}',fontsize=18) 
        
        plt.setp(ax[0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')
        
        Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
        fig.savefig(f'../pics/paper_{date.today()}/fig5-reinfect-{c}.pdf')  


def fig6_allGDP():
    cset = ['AR','AU','BE','BR','CA','FR','DE',      
          'IN','ID','IT','JP','KR','MY','MX',
          'NL','PL','RU','SA','SG','ZA','ES','SE',
          'CH','TH','TR','GB','US']
    df_out = all_output(cset)        
    df = df_out.sort_values('GDP 2021', ascending=False)
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    ax.bar(df.index, df['GDP 2021'], label='Baseline', align='center', alpha=1)
    ax.plot(df.index,df['GDP 2021 vacworse'], marker="D", label='Low vaccines', linestyle="", alpha=0.8, color="orange")
    ax.plot(df.index,df['GDP 2021 reinfect'], marker="X", label='Reinfection', linestyle="", alpha=0.8, color="r")
    #ax.plot(df.index,df['GDP 2021 3rdwave'], marker="P", linestyle="", alpha=0.8, color="green")
    
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[2], handles[0], handles[1]]
    labels = [labels[2], labels[0], labels[1]]
    ax.legend(handles,labels,loc=2)
    ax.legend(handles,labels,loc='lower left',framealpha=0 ,fontsize='x-large')
    
    plt.setp(ax.get_xticklabels(), fontsize='large')
    plt.setp(ax.get_yticklabels(), fontsize='large')
    plt.ylabel('Percent', fontsize=16)
    
    ax.set_xticklabels(["\n"*(i%2) + l for i,l in enumerate(df.index)])
    
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig6-allGDP.pdf')  

def fig6b_allGDP():
    cset = ['AR','AU','BE','BR','CA','FR','DE',      
          'IN','ID','IT','JP','KR','MY','MX',
          'NL','PL','RU','SA','SG','ZA','ES','SE',
          'CH','TH','TR','GB','US']
    df_out = all_output(cset)        
    df = df_out.sort_values('GDP 2021', ascending=False)
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    ax.bar(df.index, df['GDP 2021'], label='Baseline', align='center', alpha=1)
    ax.plot(df.index,df['GDP 2021 3rdwave'], marker="P", label='3rd wave', linestyle="", alpha=0.8, color="green")
    ax.plot(df.index,df['GDP 2021 vacworse'], marker="D", label='Low vaccines', linestyle="", alpha=0.8, color="orange")
    ax.plot(df.index,df['GDP 2021 reinfect'], marker="X", label='Reinfection', linestyle="", alpha=0.8, color="r")
    
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[0], labels[1], labels[2]]
    ax.legend(handles,labels,loc=2)
    ax.legend(handles,labels,loc='lower left',framealpha=0 ,fontsize='x-large')
    
    plt.setp(ax.get_xticklabels(), fontsize='large')
    plt.setp(ax.get_yticklabels(), fontsize='large')
    plt.ylabel('Percent', fontsize=16)
    
    ax.set_xticklabels(["\n"*(i%2) + l for i,l in enumerate(df.index)])
    
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig6b-allGDP.pdf')  

def scatter1(x,y,xlab,ylab,df):
    x1 = df[x]
    y1 = df[y]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(x1,y1,marker='o',facecolors='none', edgecolors='none')
    for i, label in enumerate(df.index):
        ax.annotate(label, (x1.iloc[i], y1.iloc[i]), size=16)
    ax.plot(np.unique(x1),
            np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)),
            color='black')
    ax.set_xlabel(xlab,size=20)
    ax.set_ylabel(ylab,size=20)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    return fig, ax

def fig7_mob2GDP():
    df = pd.read_pickle('../data/y_mob.pkl')
    fig, ax = scatter1('Mob_avg','Growth','Average mobility in 2020','Growth revisions in 2020',df)
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig7-mob2GDP.pdf')  

def scatter2(x,y,xlab,ylab,df):
    x1 = df[x]
    y1 = df[y]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(x1,y1,marker='o',facecolors='none', edgecolors='none')
    for i, label in enumerate(df.index):
        if label in ['AR','BR','ID','IN','KR','MY','MX','PL','RU','SA','SG','TH','TR','ZA']:
            color1 = 'red'
        else:
            color1 = 'black'
        ax.annotate(label, (x1.iloc[i], y1.iloc[i]), size=16, color=color1)
    ax.annotate('$45\degree$ line', (-1,0.5), size=16)
    ax.axline([0, 0], [1, 1], color='black', linestyle='--')
    ax.set_xlabel(xlab,size=20)
    ax.set_ylabel(ylab,size=20)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    return fig, ax

def fig8_scatterscenarios(cset=['US','DE']):
    df_out = all_output(cset)
    fig, ax = scatter2('GDP 2021 reinfect','GDP 2021 3rdwave','Output losses, reinfection','Output losses, third wave',df_out)
    Path(f'../pics/paper_{date.today()}').mkdir(exist_ok=True)
    fig.savefig(f'../pics/paper_{date.today()}/fig8-scenarios.pdf')  
    
def fig_update_multi(cset=['US','DE','GB','FR'],fname='multi'):
    transpa = 0.0
    color2 = 'dodgerblue'
    cvec = ['firebrick', 'darkorange', 'cornflowerblue', 'seagreen']
    cn = 0
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), constrained_layout=True)
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()
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

# Generate all figures in the paper
fig1()
fig2_baseline(cset)
fig2_baseline_multi(['US','DE','GB','FR'],'ADV')
fig2_baseline_multi(['BR','IN','KR','ZA'],'EME')
fig3_spike(cset)
fig4_vac(cset)
fig5_reinfect(cset)
fig6_allGDP()     
fig6b_allGDP()
fig7_mob2GDP()
fig8_scatterscenarios(cset)

fig_update_multi(['US','DE','GB','FR'],'ADV')
fig_update_multi(['BR','IN','KR','ZA'],'EME')

cset2 = ['US','DE','GB','FR','ES','IT','CH','JP',
         'BR','MX','IN','KR','ZA']
df_update = update_table(cset2)
print(df_update.to_markdown(tablefmt="grid"))

# Save estimated parameters and simulation results in spreadsheets 
run_baseline(cset)
save_results(cset)



toc = time.perf_counter()
print(f'Analysis complete in {(toc - tic)/60:0.2f} minutes')