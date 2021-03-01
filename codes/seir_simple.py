# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:27:49 2020

@author: Phurichai Rungcharoenkitkul

"""

import pickle
import pandas as pd
import numpy as np
from country import country
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import brute
from scipy.interpolate import interp1d
from scipy.ndimage.filters import uniform_filter1d
import psutil
from functools import partial
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
import pdb
from datetime import date, datetime, timedelta
import time
from pathlib import Path
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.patches as mpatches
import country_converter as coco 
import math
import seaborn as sns
# --------------------------------------------------------
# Global variables, chosen cohorts of data and estimates
# --------------------------------------------------------
from param_simple import *

# ----------------------
# Main class
# ----------------------

class solveCovid:  
    def __init__(self,iso2: str):  # eg 'US'
        self.iso2 = iso2
        # Policy strategies for forecast 
        self.policy = 'optim' # ['optim', 'linear']
        self.phi_option = 'fit' # ['fit','exo']: Fit phi to latest data or specify as exogenous
        self.phi_exo = 2.5e-9 # weight on mobility in social welfare function
        self.phi_min = 1e-13 # Lowerbound for phi - authorities care about output
        self.pdth_min = 0.005 # Lowerbound on death probability - countries with very few cases still think there is death probability
        # Infection rate model for forecast 
        self.gamma_tilde_model = 'AR1' # ['AR1','AR2','shock']
        self.gamma_shock_length = 10 # Shock gamma_tilde for x days  
        self.gamma_shock_depth = 0.5 # Daily increment of gamma
        self.T1_date = T1_date
        self.default_init_single = default_init_single
        self.default_bounds_single = default_bounds_single
        # Vaccine assumptions
        self.vac_assump = 'vac_base' # ['vac_base','vac_worse']
        self.effi_one = 0.5 # efficacy after one dose
        self.effi_two = 0.95 # efficacy after two doses
        self.target_weight = 0.8 # How targeted vaccine distribution is (1 = sequenced from eldest to youngest, 0 is random)
        self.vac_base_use_A = 1 # Baseline: Group A (already started): % of contracted dosages deployed by December 2021
        self.vac_base_start_B = '2021-03-31' # Baseline: Group B (hasn't started): first date of vaccination 
        self.vac_base_use_B = 0.75 # Baseline: Group B: % of contracted dosages deployed by December 2021
        self.vac_worse_use_A = 0.3 # Worse A: Use by end of 2021
        self.vac_worse_start_B = '2021-06-30' # Worse B: Starting date
        self.vac_worse_use_B = 0.3 # Worse B: Use by end of 2021
        self.vac_better_use_A = 1.3
        self.vac_better_start_B = '2021-03-31'
        self.vac_better_use_B = 1
        # Reinfection and loss of immunity
        self.reinfect = 'immune' # ['immune','reinfect']
        self.r_re1 = np.log(2)/10000 # Baseline: lost immunity after 3 years
        self.r_re2 = np.log(2)/60 # Lost immunity after 60 days, approx 1% of V+R lose immunity each day

    # --------------- 1. Preliminary: Get the data ------------------------
    def prelim(self):
        iso2 = self.iso2
        self.N = df1.fillna(method='ffill')['population'][iso2].iloc[-1]
        df2 = df1.iloc[:,df1.columns.get_level_values(1)==iso2][[
                'total_cases','total_deaths','new_cases','new_deaths',
                'google_smooth','vac_total','vac_partial',
                'vac_fully']][df1['total_cases'][iso2] > virus_thres] 
        df2['vac_total'] = df2['vac_total'].interpolate()
        df2['vac_partial'] = df2['vac_partial'].interpolate()
        df2['vac_fully'] = df2['vac_fully'].interpolate()
        if np.isnan(df2['vac_partial'].iloc[-1].values[0]) and ~np.isnan(df2['vac_total'].iloc[-1].values[0]):
            df2['vac_partial'] = 0.8 * df2['vac_total'] # If no data on breakdowns exist, do manual approximation
            df2['vac_fully'] = 0.2 * df2['vac_total']
        df2 = df2.fillna(0) # Replace NaN by 0 - deaths and vaccinations
        df2 = df2.droplevel('iso2',axis=1)
        PopulationI = df2['total_cases'][0]
        PopulationD = df2['total_deaths'][0]
        if PopulationD==0:
            PopulationD = 0
            PopulationR = 5
        else:
            PopulationR = PopulationD * 5
        PopulationCI = PopulationI - PopulationD - PopulationR # Undetected and infectious cases
        self.cases_data_fit = df2['total_cases'].tolist()
        self.deaths_data_fit = df2['total_deaths'].tolist()        
        self.newcases_data_fit = df2['new_cases'].tolist()
        self.newdeaths_data_fit = df2['new_deaths'].tolist() 
        self.balance = self.cases_data_fit[-1] / max(self.deaths_data_fit[-1], 10) / 3
        date_day_since100 = pd.to_datetime(df2.index[0])
        self.maxT = (default_maxT - date_day_since100).days + 1
        self.T1 = (self.T1_date - date_day_since100).days + 1
        self.mobility_vec = df2['google_smooth'].values
        self.T = len(df2)
        self.t_cases = np.arange(0,self.T)
        self.mobility_interp = interp1d(self.t_cases,self.mobility_vec,bounds_error=False,fill_value=0.,kind='cubic')
        self.GLOBAL_PARAMS = (self.N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v)        
        self.gamma_0_days = 1 # average of gamma_t during first n days becomes the target 
        # Compute vaccination parameters
        self.vac_partial = df2['vac_partial'].values
        self.vac_fully = df2['vac_fully'].values     
        self.vac_contracted = 1000*df_vac.loc[iso2]['No. of people covered (thousands)']/self.N
        df2['V_'] = self.N * (self.effi_one*df2['vac_partial']
                    + self.effi_two*df2['vac_fully'])/100 # V = expected number of effectively vaccinated persons
        ix = pd.date_range(start=df2.index[0], end=default_maxT, freq='D') # Expand time-sample, to include forecast later
        df_v = df2.reindex(ix)
        # Vaccination assumptions  
        if self.vac_assump == 'vac_base':              
            if df2['V_'][-1] > 0: # If already vaccinating, assume self.vac_base_use_A % of orders are delivered and used by 2021 (if supply>100%, translates into faster distribution)            
                df_v['V_'].loc['2021-12-31'] = self.vac_base_use_A * self.N * self.vac_contracted 
            elif df2['V_'][-1] == 0: # If has not started, assume starting by xxx and fill orders by xxx
                df_v['V_'].loc[self.vac_base_start_B] = 100 # 100 = assumed number of effectively vaccinated on first day
                df_v['V_'].loc['2021-12-31'] = self.vac_base_use_B*( self.N * self.vac_contracted) # partial orders filled by year end
        elif self.vac_assump == 'vac_worse':
            if df2['V_'][-1] > 0:
                df_v['V_'].loc['2021-12-31'] = self.vac_worse_use_A * self.N * self.vac_contracted # Only half of contracted doses are used
            elif df2['V_'][-1] == 0:
                df_v['V_'].loc[self.vac_worse_start_B] = 100 # Start delayed 
                df_v['V_'].loc['2021-12-31'] = self.vac_worse_use_B*(self.N * self.vac_contracted) # Half of (smaller orders) are used
        elif self.vac_assump == 'vac_better':
            if df2['V_'][-1]>0:
                df_v['V_'].loc['2021-12-31'] = self.vac_better_use_A*self.N*self.vac_contracted
            elif df2['V_'][-1] == 0: # If has not started, assume starting by xxx and fill orders by xxx
                df_v['V_'].loc[self.vac_better_start_B] = 100 # 100 = assumed number of effectively vaccinated on first day
                df_v['V_'].loc['2021-12-31'] = self.vac_better_use_B*( self.N * self.vac_contracted) # partial orders filled by year end        
        df_v['V_'] = df_v['V_'].interpolate()
        df_v['V_'] = df_v['V_'].clip(0,self.N)
        self.df2 = df2
        self.df_v = df_v
        print(f'Data preparation for {iso2} done')
        
    # --------------------------3 . SEIR model ------------------
    def step_seir(self, t, x, gamma_t, p_dth) -> list:
        """
        SEIR model building on DELPHI v.3
        Features 16 distinct states, taking into account undetected, deaths, hospitalized and
        recovered
        [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 UD, 7 DHD, 8 DQD, 9 R, 10 D, 
         11 TH, 12 DVR,13 DVD, 14 DD, 15 DT, 16 V]
        """
        S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT, V = x       
        r_v = self.df_v['V_'].iloc[t+1] - self.df_v['V_'].iloc[t]
        # Reinfection parameters
        if self.reinfect == 'immune':
            r_re = self.r_re1
        elif self.reinfect == 'reinfect':
            if t <= self.T:
                r_re = self.r_re1
            else:
                r_re = self.r_re2
        # Main equations
        S1 = S - gamma_t * S * I / self.N + r_re*R +r_re*V - r_v
        if S1 < 0: # Vaccination reaches saturating point
            S1 = 0
            r_v = S - gamma_t * S * I / self.N + r_re*R +r_re*V
        E1 = E + gamma_t * S * I / self.N - r_i * E
        I1 = I + r_i * E - r_d * I
        AR1 = AR + r_d * (1 - p_dth) * (1 - p_d) * I - r_ri * AR
        DHR1 = DHR + r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
        DQR1 = DQR + r_d * (1 - p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
        AD1 = AD + r_d * p_dth * (1 - p_d) * I - r_dth * AD
        DHD1 = DHD + r_d * p_dth * p_d * p_h * I - r_dth * DHD
        DQD1 = DQD + r_d * p_dth * p_d * (1 - p_h) * I - r_dth * DQD
        R1 = R + r_ri * (AR + DQR) + r_rh * DHR - r_re*R
        D1 = D + r_dth * (AD + DQD + DHD)
        # Helper states 
        TH1 = TH + r_d * p_d * p_h * I
        DVR1 = DVR + r_d * (1 - p_dth) * p_d * p_h * p_v * I - r_rv * DVR
        DVD1 = DVD + r_d * p_dth * p_d * p_h * p_v * I - r_dth * DVD
        DD1 = DD + r_dth * (DHD + DQD)
        DT1 = DT + r_d * p_d * I
        V1 = V + r_v -r_re*V
        x1 = [S1, E1, I1, AR1, DHR1, DQR1, AD1, DHD1, DQD1,
              R1, D1, TH1, DVR1, DVD1, DD1, DT1, V1]
        return x1
    
    # ------------------ X. Construct initial conditions
    def initial_states_func(self,k):
        N, PopulationCI, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v = self.GLOBAL_PARAMS      
        p_dth0 = self.newdeaths_data_fit[0]/(r_dth*PopulationCI) # Set p_dth0 to match D1-D0 to newdeaths_data_fit
        E_0 = PopulationCI / p_d * k
        I_0 = PopulationCI / p_d * k
        UR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth0)
        DHR_0 = (PopulationCI * p_h) * (1 - p_dth0)
        DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth0)
        UD_0 = (PopulationCI / p_d - PopulationCI) * p_dth0
        DHD_0 = PopulationCI * p_h * p_dth0
        DQD_0 = PopulationCI * (1 - p_h) * p_dth0
        R_0 = PopulationR / p_d
        D_0 = PopulationD / p_d
        S_0 = N - (E_0 +I_0 +UR_0 +DHR_0 +DQR_0 +UD_0 +DHD_0 +DQD_0 +R_0 +D_0)
        TH_0 = PopulationCI * p_h
        DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth0)
        DVD_0 = (PopulationCI * p_h * p_v) * p_dth0
        DD_0 = PopulationD
        DT_0 = PopulationI
        V_0 = 0
        x_init = [
            S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0, DQD_0, R_0,
            D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0, V_0
            ] 
        return x_init          
    
    # Find k=k1,k2 that matches gamma_0 to 2.08 (R0=6 equivalent)   
    def loss_gamma0(self,k):
        newcases = np.array(self.newcases_data_fit)
        newdeaths = np.array(self.newdeaths_data_fit)
        newcases_sm = uniform_filter1d(newcases, size=21, mode='nearest')
        newdeaths_sm = uniform_filter1d(newdeaths, size=21, mode='nearest')
        gamma_t_vec = []  
        
        x_init = self.initial_states_func(k)
        (S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0, DQD_0, R_0, 
         D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0, V_0) = x_init
        
        newcases_sm2 = np.append(newcases_sm, newcases_sm[-2:]) # Extend the list for forward projection below
        newdeaths_sm2 = np.append(newdeaths_sm, newdeaths_sm[-1])
        x_0 = x_init.copy()
        for t in range(self.gamma_0_days): # Target first n days
            gamma_t = (newcases_sm2[t+2]/(r_d*p_d) - (1-r_d)**2 *I_0 - r_i*(2-r_d-r_i)*E_0 )*self.N/(r_i*S_0*I_0)
            p_dth = (newdeaths_sm2[t+1] - r_dth*(1-r_dth)*(DHD_0 + DQD_0))/(r_dth*r_d*p_d*I_0)
            gamma_t = np.clip(gamma_t, 0.01, 10)
            p_dth = np.clip(p_dth,0,1) # Probability limit [0,1]
            x_1 = self.step_seir(t, x_0, gamma_t, p_dth)
            x_0 = x_1
            gamma_t_vec.append(gamma_t)
        gamma_0 = np.mean(gamma_t_vec)       
        loss = (gamma_0 - (r_d*6) )**2 # gamma_0 equivalent to R0=6 is 2.08
        return loss
    
    def fit_gamma0(self):
        output = dual_annealing(
            self.loss_gamma0,
            x0 = [5],
            bounds = [(1,50)],
            )
        k_star = output.x    
        return k_star    
    
    def get_initial_conditions(self):
        if Path(f'../params/param_fixed/kstar.csv').exists():
            df = pd.read_csv(f'../params/param_fixed/kstar.csv')
            kstar = df[self.iso2].values[0]
        else:
            kstar = self.fit_gamma0()[0] # find kstar that matches gamma_0 to target
        x_init = self.initial_states_func(kstar)
        return x_init
    
    # -------------------- x. Implied gamma_t and pdth_t in-sample -------------------
    def gamma_t_compute(self):
        newcases = np.array(self.newcases_data_fit)
        newdeaths = np.array(self.newdeaths_data_fit)
        newcases_sm = uniform_filter1d(newcases, size=21, mode='nearest')
        newdeaths_sm = uniform_filter1d(newdeaths, size=21, mode='nearest')
        gamma_t_vec = []
        p_dth_vec = []
        x_init = self.get_initial_conditions() 
        S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0, R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0, V_0 = x_init
        S_vec = [S_0]
        E_vec = [E_0]
        I_vec = [I_0]
        DT_vec = [DT_0]
        DD_vec = [DD_0]
        newcases_sm2 = np.append(newcases_sm, newcases_sm[-2:]) # Extend the list for forward projection below
        newdeaths_sm2 = np.append(newdeaths_sm, newdeaths_sm[-1])
        x_0 = x_init.copy()
        for t in range(len(newcases)):
            # Work backwards to compute 'exact' gamma_t and p_dth
            gamma_t = (newcases_sm2[t+2]/(r_d*p_d) - (1-r_d)**2 *I_0 - r_i*(2-r_d-r_i)*E_0 )*self.N/(r_i*S_0*I_0)
            p_dth = (newdeaths_sm2[t+1] - r_dth*(1-r_dth)*(DHD_0 + DQD_0))/(r_dth*r_d*p_d*I_0)
            gamma_t = np.clip(gamma_t, 0.01, 10)
            p_dth = np.clip(p_dth,0,1) # Probability limit [0,1]
            x_1 = self.step_seir(t, x_0, gamma_t, p_dth)
            S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0, R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0, V_0 = x_1
            x_0 = x_1
            gamma_t_vec.append(gamma_t)
            p_dth_vec.append(p_dth)
            S_vec.append(S_0)
            I_vec.append(I_0)
            E_vec.append(E_0)
            DT_vec.append(DT_0)
            DD_vec.append(DD_0)
        self.df2['gamma_t'] = gamma_t_vec
        self.df2['pdth_t'] = p_dth_vec
        self.S_vec = S_vec # In-sample estmates, useful for phi calculation later on
        self.I_vec = I_vec
        # gamma_t_sm = uniform_filter1d(gamma_t_vec, size=6, mode='nearest')
        # self.df2['gamma_sm'] = gamma_t_sm
        return gamma_t_vec, p_dth_vec
        
    # -------------------- x. Estimating the model -----------
    
    def gamma_func(self, params):
        m_t = self.df2['google_smooth'].values
        tvec = np.arange(len(m_t))
        beta0, beta1 = params
        gamma_vec = beta0*np.exp(beta1* m_t)                      
        return gamma_vec
    
    def loss_betas(self, params) -> float:
        gamma_model = self.gamma_func(params)
        loss = sum( (self.df2['gamma_t'].values[:len(gamma_model)] - gamma_model)**2 )
        return loss
    
    def fitmodel(self):
        #    A. Fit beta0 and beta1 
        x0 = self.default_init_single
        bounds_0 = self.default_bounds_single
        output = dual_annealing(
            self.loss_betas,
            x0 = x0,
            bounds = bounds_0,
            )
        best_betas = output.x    
        self.best_betas = best_betas
        #   B. Fit the residual (gamma_tilde) to AR models 
        m_t = self.df2['google_smooth'].values
        tvec = np.arange(len(self.df2))
        beta0, beta1 = self.best_betas
        self.df2['gamma_mob'] = beta0*np.exp(beta1* m_t)
        self.df2['gamma_tilde'] = self.df2['gamma_t'] - self.df2['gamma_mob']
        self.df2['gamma_tilde_sm'] = uniform_filter1d(self.df2['gamma_tilde'],
                                          size=21, mode='reflect')
        self.df2['gamma_tilde_resid'] = self.df2['gamma_tilde'] - self.df2['gamma_tilde_sm']
        y = self.df2['gamma_tilde_sm']
        self.df2['gamma_tilde_sm_lag1'] = self.df2['gamma_tilde_sm'].shift(1) # No constant term
        self.df2['gamma_tilde_sm_lag2'] = self.df2['gamma_tilde_sm'].shift(2)
        reg_AR1 = sm.OLS(y,self.df2['gamma_tilde_sm_lag1'],missing='drop').fit()
        reg_AR2 = sm.OLS(y,self.df2[['gamma_tilde_sm_lag1','gamma_tilde_sm_lag2']],missing='drop').fit()
        best_rho1 = reg_AR1.params[0]
        best_rho1 = np.clip(best_rho1, 0.1, 0.99) #Assume stationarity
        best_rho2 = reg_AR2.params[:]
        best_params = np.array([beta0, beta1, best_rho1, best_rho2[0], best_rho2[1]])
        self.best_rho1 = best_rho1
        self.best_rho2 = best_rho2
        self.best_params = best_params
        #   C. Empirically fit phi for optimal policy to last observation 
        if self.phi_option == 'fit':
            m = self.df2['google_smooth'][-15:].mean() # Take average of last 15 days to smooth volatility
            s = self.S_vec[-1]/self.N
            i = self.I_vec[-1]/self.N
            gamma_tilde = self.df2['gamma_tilde'][-1]
            pdth = self.df2['pdth_t'][-1]
            pdth = max(pdth, self.pdth_min) # Get around cases where pdth=0 for countries with very few cases
            LHS1 = pdth*r_d*i*s*(beta0*beta1*np.exp(beta1*m))
            LHS2 = pdth*r_d*i*(1 - r_d + s*(gamma_tilde + beta0*np.exp(beta1*m)))
            phi = -(LHS1 * LHS2)/m
            self.phi = max(phi, self.phi_min)
        elif self.phi_option == 'exo':
            self.phi = self.phi_exo
        return best_params      
    
    # ------------------ x. Forecasts ---------------------------
    def step_gamma_tilde(self, gamma_tilde_lag1, gamma_tilde_lag2, model='AR1'):
        if model =='AR1':
            return self.best_rho1*gamma_tilde_lag1
        elif model =='AR2':
            return self.best_rho2[0]*gamma_tilde_lag1 + self.best_rho2[1]*gamma_tilde_lag2
    def mobility_choice(self,x,gamma_tilde,pdth):
        if self.policy == 'constant':
            mob = self.poparam_constant
        elif self.policy == 'linear-I': # Respond linearly to infection level
            mob = self.poparam_linear_I[0] + self.poparam_linear_I[1]*x[2] 
        elif self.policy == 'linear-dI': # Respond to new infections
            dI = r_i*x[1] - r_d*x[2] # x[1]=E, x[2]=I
            mob = self.poparam_linear_dI[0] + self.poparam_linear_dI[1]*dI
        elif self.policy == 'optim': # Analytical optimal policy based on simplified model and quadratic losses
            beta0 = self.best_params[0]
            beta1 = self.best_params[1]
            phi = self.phi
            s = x[0]/self.N
            i = x[2]/self.N
            m_set = np.linspace(-1,0,101)
            RHS = -phi*m_set
            LHS1 = pdth*r_d*i*s*(beta0*beta1*np.exp(beta1*m_set))
            LHS2 = pdth*r_d*i*(1 - r_d + s*(gamma_tilde + beta0*np.exp(beta1*m_set)))
            LHS = LHS1 * LHS2
            m_id = np.argmin(np.abs(RHS-LHS))
            mob = m_set[m_id]
        return mob
    def fatality_factor(self,V): # Factor to adjust 'base' fatality prob
        idx = (f_table[self.iso2]['vaccine_%'] - V/self.N).abs().argmin() # Find idx to look up in fatality table
        factor = f_table[self.iso2]['fatality_ratio'][idx] 
        return factor        
    def sim_seir(self):
        df2 = self.df2
        ix = pd.date_range(start=df2.index[0], end=default_maxT, freq='D') # Expand time-sample, to include forecast later
        df3 = df2.reindex(ix)
        x_init = self.get_initial_conditions() 
        x_data = np.array(x_init)
        gamma_tilde_fc = self.df2['gamma_tilde'].values
        gamma_tilde_sm_fc = self.df2['gamma_tilde_sm'].values
        pdth_t_targ = [] # Death prob when vaccines are targeted
        pdth_t_base = [] # Base death prob if vaccines are given randomly
        pdth_t_fc = self.df2['pdth_t'].values
        pdth_t_base_fc = pdth_t_fc.copy()
        gamma_mob_fc = self.df2['gamma_mob'].values
        mob_fc = self.df2['google_smooth'].values
        # Load parameters
        if hasattr(self, 'best_params'):
            beta0, beta1, rho, rhos_1, rhos_2 = self.best_params
        else:
            df_param = pd.read_csv(f'../params/{param_load_folder}/param_est.csv')
            beta0, beta1, rho, rhos_1, rhos_2 = df_param[self.iso2]

        for t in range(self.maxT):
            factor = self.fatality_factor(x_init[-1])
            eta = self.target_weight
            if t<len(self.df2): # In sample
                pdth_t = pdth_t_fc[t]
                pdth_base = pdth_t/(eta*factor + 1-eta)
                pdth_targ = factor*pdth_base        
            if t>len(self.df2)-1: # Out of sample   
                # Death probability
                pdth_base = pdth_t_base[-1] # Martingale death rate (could change)
                pdth_base = max(pdth_base, self.pdth_min) # To get around pdth=0 for countries with very few cases
                pdth_t = (eta*factor + 1-eta)*pdth_base
                pdth_targ = factor*pdth_base
                # Gamma_tilde
                if self.gamma_tilde_model == 'AR1':
                    gamma_tilde = rho*gamma_tilde_sm_fc[t-1]
                elif self.gamma_tilde_model == 'AR2':
                    gamma_tilde = rhos_1*gamma_tilde_sm_fc[t-1] + rhos_2*gamma_tilde_sm_fc[t-2]
                elif self.gamma_tilde_model =='shock':
                    if t < len(self.df2) + self.gamma_shock_length:                       
                        gamma_tilde = gamma_tilde_sm_fc[len(self.df2)-1] + self.gamma_shock_depth
                    else:
                        gamma_tilde = rho*gamma_tilde_sm_fc[t-1]
                # Mobility and overall gamma_t
                mob_t = self.mobility_choice(x_init, gamma_tilde, pdth_t)
                mob_t = max(mob_t, max_lockdown)
                gamma_mob_t = beta0*np.exp(beta1*mob_t) 
                gamma_t = gamma_tilde + gamma_mob_t
                # Append to data array
                gamma_tilde_sm_fc = np.append(gamma_tilde_sm_fc, gamma_tilde)
                gamma_tilde_fc = np.append(gamma_tilde_fc, gamma_tilde)
                gamma_mob_fc = np.append(gamma_mob_fc, gamma_mob_t)
                mob_fc = np.append(mob_fc, mob_t)
                pdth_t_fc = np.append(pdth_t_fc, pdth_t)
            pdth_t_base.append(pdth_base)
            pdth_t_targ.append(pdth_targ)
            # For in sample, use 'true' inputs
            gamma_t = gamma_tilde_fc[t] + gamma_mob_fc[t]
            p_dth = pdth_t_fc[t]   
            if t < range(self.maxT)[-1]: # Stop forecasting at the final period
                x_next = self.step_seir(t, x_init, gamma_t, p_dth)
                x_data = np.vstack((x_data, np.array(x_next)))
                x_init = x_next
        # Fill dataframe
        col_temp = ['S', 'E', 'I', 'AR', 'DHR', 'DQR', 'AD', 'DHD', 'DQD', 'R', 'D', 'TH', 'DVR', 'DVD', 'DD', 'DT', 'V']
        df4 = pd.DataFrame(x_data, columns=col_temp, index=df3.index)
        df3 = df3.merge(df4, how='left', left_index=True, right_index=True)
        df3['gamma_tilde_fc'] = gamma_tilde_fc
        df3['gamma_mob_fc'] = gamma_mob_fc
        df3['gamma_t_fc'] = df3['gamma_tilde_fc'] + df3['gamma_mob_fc']
        df3['mob_fc'] = mob_fc
        df3['pdth_t_fc'] = pdth_t_fc
        df3['pdth_t_base'] = np.array(pdth_t_base)
        df3['pdth_t_targ'] = np.array(pdth_t_targ)
        df3[['S_N','I_N','DT_N','DD_N','V_N']] = df3[['S','I','DT','DD','V']]/self.N
        self.df3 = df3
        return df3
            
    # ------------------ 5. Predict and plot ---------------------
    def plot_all(self, saveplot=False):
        df = self.df3
        transpa = 0.0
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8), constrained_layout=True)

        # df_bar = df_bar0[['GDP lost','Total deaths']]
        # df_bar.plot(kind='bar', ax=ax[1,2], secondary_y='Total deaths', rot=0, legend=False)
        # ax[1,2].set_ylabel('percent')
        # ax[1,2].right_ax.set_ylabel('per million')
        # ax[1,2].set_title('Losses of lives and output',fontsize='x-large')
        # L = [mpatches.Patch(color=c, label=col) 
        #       for col,c in zip( ('GDP loss','Deaths (rhs)'), plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        # ax[1,2] = plt.legend(handles=L, loc=1, framealpha=transpa)
        
        ax[0,0].plot(df.index, 100*df['total_cases']/self.N, linewidth = 3, label='Case data', color='blue')
        ax[0,0].plot(df.index, 100*df['DT']/self.N, label='$DT_t$', color='red')
        ax[0,0].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,0].set_title('Cases',fontsize='x-large')
        ax[0,0].set(ylabel = '% of population')
        ax2 = ax[0,0].twinx()
        ax2.plot(df.index, 100*df['I']/self.N, label='$I_t$ (rhs)',color='green',linestyle='--')
        lines, labels = ax[0,0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right', framealpha=transpa,fontsize='x-large')
        #ax2.set(ylabel='% of population')      
        
        ax[0,1].plot(df.index, 100*df['total_deaths']/self.N, linewidth = 3, label='Death data', color='blue')
        ax[0,1].plot(df.index, 100*df['DD']/self.N, label='$DD_t$', color='red')
        ax[0,1].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,1].set_title('Deaths',fontsize='x-large')
        ax[0,1].set(ylabel='% of population')
        ax[0,1].legend(loc='best', framealpha=transpa ,fontsize='x-large')
        
        ax[0,2].plot(df.index, 100*df['S']/self.N, label='$S_t$',color='red')
        ax[0,2].plot(df.index, 100*df['V']/self.N, label='$V_t$',color='red',linestyle=':')
        ax[0,2].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,2].set_title('Susceptible & vaccinated',fontsize='x-large')
        ax[0,2].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        ax[0,2].set(ylabel='% of population')
        
        ax[1,0].plot(df.index, df['gamma_t'], label=r'$\gamma_t$',color='red')
        ax[1,0].plot(df.index, df['gamma_mob'], label=r'$\gamma^{m}_t$', color ='blue')
        ax[1,0].plot(df.index, df['gamma_tilde'], label=r'$\gamma^{d}$', color='orange')
        ax[1,0].plot(df.index, df['gamma_t_fc'], color='red',linestyle=':')
        ax[1,0].plot(df.index, df['gamma_mob_fc'], color ='blue',linestyle=':')
        ax[1,0].plot(df.index, df['gamma_tilde_fc'], color='orange',linestyle=':')
        ax[1,0].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,0].set_title('Infection rate',fontsize='x-large')
        ax[1,0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        
        ax[1,1].plot(df.index, 100*df['google_smooth'], linewidth = 3, label='Google mobility', color='blue')
        ax[1,1].plot(df.index, 100*df['mob_fc'], label='Model', color='red')
        ax[1,1].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,1].legend(loc=0,framealpha=transpa ,fontsize='x-large')
        ax[1,1].set_title('Activity',fontsize='x-large')
        ax[1,1].set(ylabel='% deviations from norm')
        
        ax[1,2].plot(df.index, 100*df['pdth_t'], label='Death probability', linewidth=3, color='blue')
        ax[1,2].plot(df.index, 100*df['pdth_t_fc'], color='black', label='Forecast')
        ax[1,2].plot(df.index, 100*df['pdth_t_base'], color='black', linestyle='dashed', label='Random vaccines')
        ax[1,2].plot(df.index, 100*df['pdth_t_targ'], color='black', linestyle=':', label='Targeted vaccines')
        ax[1,2].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,2].legend(loc=0,framealpha=transpa ,fontsize='x-large')
        ax[1,2].set_title('Death probability',fontsize='x-large')
        ax[1,2].set(ylabel='%')
    
        plt.setp(ax[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[0,2].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,2].get_xticklabels(), rotation=30, horizontalalignment='right')
        cname = coco.convert(names=self.iso2,to='name_short')
        fig.suptitle(f'{cname}-{self.vac_assump}-{self.reinfect}',fontsize='xx-large') 

        if saveplot:
            Path(f'../pics/fig_{date.today()}').mkdir(exist_ok=True)
            fig.savefig(f'../pics/fig_{date.today()}/{self.iso2}-{self.policy}-{self.gamma_tilde_model}-{self.vac_assump}-{self.reinfect}.png')        
        return fig

    def plot_portrait(self, saveplot=False):
        df = self.df3
        transpa = 0.0
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,12), constrained_layout=True)
       
        ax[0,0].plot(df.index, 100*df['total_cases']/self.N, linewidth = 3, label='Case data', color='blue')
        ax[0,0].plot(df.index, 100*df['DT']/self.N, label='$DT_t$', color='red')
        ax[0,0].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,0].set_title('Cases',fontsize='x-large')
        ax[0,0].set(ylabel = '% of population')
        ax2 = ax[0,0].twinx()
        ax2.plot(df.index, 100*df['I']/self.N, label='$I_t$ (rhs)',color='green',linestyle='--')
        ax2.grid(None)
        lines, labels = ax[0,0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right', framealpha=transpa,fontsize='x-large')
        #ax2.set(ylabel='% of population')      
        
        ax[0,1].plot(df.index, 100*df['total_deaths']/self.N, linewidth = 3, label='Death data', color='blue')
        ax[0,1].plot(df.index, 100*df['DD']/self.N, label='$DD_t$', color='red')
        ax[0,1].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,1].set_title('Deaths',fontsize='x-large')
        ax[0,1].set(ylabel='% of population')
        ax[0,1].legend(loc='best', framealpha=transpa ,fontsize='x-large')
        
        ax[1,0].plot(df.index, 100*df['S']/self.N, label='$S_t$',color='red')
        ax[1,0].plot(df.index, 100*df['V']/self.N, label='$V_t$',color='red',linestyle=':')
        ax[1,0].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,0].set_title('Susceptible & vaccinated',fontsize='x-large')
        ax[1,0].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        ax[1,0].set(ylabel='% of population')
        
        ax[1,1].plot(df.index, df['gamma_t'], label=r'$\gamma_t$',color='red')
        ax[1,1].plot(df.index, df['gamma_mob'], label=r'$\gamma^{m}_t$', color ='blue')
        ax[1,1].plot(df.index, df['gamma_tilde'], label=r'$\gamma^{d}$', color='orange')
        ax[1,1].plot(df.index, df['gamma_t_fc'], color='red',linestyle=':')
        ax[1,1].plot(df.index, df['gamma_mob_fc'], color ='blue',linestyle=':')
        ax[1,1].plot(df.index, df['gamma_tilde_fc'], color='orange',linestyle=':')
        ax[1,1].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,1].set_title('Infection rate',fontsize='x-large')
        ax[1,1].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        
        ax[2,0].plot(df.index, 100*df['google_smooth'], linewidth = 3, label='Google mobility', color='blue')
        ax[2,0].plot(df.index, 100*df['mob_fc'], label='Model', color='red')
        ax[2,0].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[2,0].legend(loc=0,framealpha=transpa ,fontsize='x-large')
        ax[2,0].set_title('Mobility',fontsize='x-large')
        ax[2,0].set(ylabel='% deviations from norm')
        
        ax[2,1].plot(df.index, 100*df['pdth_t'], label='Death probability', linewidth=3, color='blue')
        ax[2,1].plot(df.index, 100*df['pdth_t_fc'], color='black', label='Forecast')
        ax[2,1].plot(df.index, 100*df['pdth_t_base'], color='black', linestyle='dashed', label='Random vaccines')
        ax[2,1].plot(df.index, 100*df['pdth_t_targ'], color='black', linestyle=':', label='Targeted vaccines')
        ax[2,1].axvline(df.index[self.T], linewidth = 2, color='gray', linestyle=':')
        ax[2,1].legend(loc=0,framealpha=transpa ,fontsize='x-large')
        ax[2,1].set_title('Death probability',fontsize='x-large')
        ax[2,1].set(ylabel='%')
    
        plt.setp(ax[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[2,0].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[2,1].get_xticklabels(), rotation=30, horizontalalignment='right')

        cname = coco.convert(names=self.iso2,to='name_short')
        fig.suptitle(f'{cname}',fontsize=18) 

        if saveplot:
            Path(f'../pics/fig_{date.today()}').mkdir(exist_ok=True)
            fig.savefig(f'../pics/fig_{date.today()}/Portrait-{self.iso2}-{self.policy}-{self.gamma_tilde_model}-{self.vac_assump}-{self.reinfect}.pdf')        
        return fig
    

# ---------------------------------------------
#       Calling functions
# ---------------------------------------------

# -----------------------------------------
# x.    Prelim parameters estimation
# Estimate k_star and save in file (only need to do this once)
def estimate_kstar(cset=['US']):
    dict = {'Parameter': ['kstar']}
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        kstar = tmp.fit_gamma0()
        dict[c] = kstar
    df = pd.DataFrame(dict)
    df.to_csv(f'../params/param_fixed/kstar.csv',index=False)
    return df

# -------------------------
# x. Run complete package under scenarios: estimate, forecast, plot, save
def run_baseline(cset=['US']):
    p_dict = {'Parameters': ['beta0','beta1','rho','rhos_1','rhos_2','phi']}
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        p_dict[c] = np.append(tmp.best_params, 1e9*tmp.phi)
        tmp.sim_seir()
        tmp.plot_all(saveplot='False')
        tmp.df3.to_csv(f'../output/{out_save_folder}/df3_{tmp.iso2}.csv')
    pd.DataFrame(p_dict).to_csv(f'../params/{param_save_folder}/param_est.csv',float_format='%.4f',index=False)
    
def run_gammashock(cset=['US']):
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.gamma_tilde_model = 'shock'
        tmp.sim_seir()
        tmp.plot_all(saveplot=True)        

def run_vaccines(cset=['US'],vac_assump='vac_worse'):
    for c in cset:
        tmp = solveCovid(c)
        tmp.vac_assump = vac_assump
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()
        tmp.plot_all(saveplot=True)   
    
def run_reinfect(cset=['US'],reinfect = 'reinfect'):
    for c in cset:
        tmp = solveCovid(c)
        tmp.reinfect = reinfect
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        tmp.sim_seir()
        tmp.plot_all(saveplot=True)          

def run_scenarios(cset=['US']): # Save class objects under various scenarios so we could draw plots across countries/scenarios
    p_dict = {'Parameters': ['beta0','beta1','rho','rhos_1','rhos_2','phi']}
    for c in cset:
        #Baseline
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        tmp.fitmodel()
        p_dict[c] = np.append(tmp.best_params, 1e9*tmp.phi)
        tmp.sim_seir()
        tmp.plot_all(saveplot=True)
        name = f'../output/{out_save_folder}/{c}_baseline.pkl'
        pickle.dump(tmp,open(name,'wb'))
        # Vaccines
        t_vac = solveCovid(c)
        t_vac.vac_assump = 'vac_worse'
        t_vac.prelim()
        t_vac.gamma_t_compute()
        t_vac.fitmodel()
        t_vac.sim_seir()
        t_vac.plot_all(saveplot=True)
        name = f'../output/{out_save_folder}/{c}_vacworse.pkl'
        pickle.dump(t_vac,open(name,'wb'))
        # Spikes
        t_spike = solveCovid(c)
        t_spike.prelim()
        t_spike.gamma_t_compute()
        t_spike.fitmodel()
        t_spike.gamma_tilde_model = 'shock'
        t_spike.sim_seir()
        t_spike.plot_all(saveplot=True)
        name = f'../output/{out_save_folder}/{c}_shock.pkl'
        pickle.dump(t_spike,open(name,'wb'))        
        # Reinfection
        t_reinfect = solveCovid(c)
        t_reinfect.reinfect = 'reinfect'
        t_reinfect.prelim()
        t_reinfect.gamma_t_compute()
        t_reinfect.fitmodel()
        t_reinfect.sim_seir()
        t_reinfect.plot_all(saveplot=True)
        name = f'../output/{out_save_folder}/{c}_reinfect.pkl'
        pickle.dump(t_reinfect,open(name,'wb'))
        # Better
        t_better = solveCovid(c)
        t_better.vac_assump = 'vac_better' # (a) 30% Faster vaccines
        t_better.target_weight = 0.9    # (b) More targeted
        t_better.prelim()
        t_better.gamma_t_compute()
        t_better.fitmodel()
        t_better.sim_seir()
        t_better.plot_all(saveplot=True)
        name = f'../output/{out_save_folder}/{c}_better.pkl'
        pickle.dump(t_better,open(name,'wb'))
    pd.DataFrame(p_dict).to_csv(f'../params/{param_save_folder}/param_est.csv',float_format='%.4f',index=False)
     
def save_results(cset=['US']): # Unpack pickle and save all results into an excel
    with pd.ExcelWriter(f'../output/{out_save_folder}/output_all.xlsx') as writer: 
        for c in cset:
            print(f'Loading pickle for {c}')
            tmp = pickle.load(open(f'../output/{out_load_folder}/{c}_baseline.pkl','rb'))
            t_vac = pickle.load(open(f'../output/{out_load_folder}/{c}_vacworse.pkl','rb'))
            t_spike = pickle.load(open(f'../output/{out_load_folder}/{c}_shock.pkl','rb'))
            t_reinfect = pickle.load(open(f'../output/{out_load_folder}/{c}_reinfect.pkl','rb'))
            t_better = pickle.load(open(f'../output/{out_load_folder}/{c}_better.pkl','rb'))
            tmp.df3.to_excel(writer, sheet_name=f'{c}_base')
            t_vac.df3.to_excel(writer, sheet_name=f'{c}_vacworse')
            t_spike.df3.to_excel(writer, sheet_name=f'{c}_shock')
            t_reinfect.df3.to_excel(writer, sheet_name=f'{c}_reinfect')
            t_better.df3.to_excel(writer, sheet_name=f'{c}_better')

# ---------------------------------------------------
# x. Plotting functions        

#  *****  Utilities *****
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
def scatter2(x,y,x2,y2,xlab,ylab,df):
    x1 = df[x]
    y1 = df[y]
    x2 = df[x2]
    y2 = df[y2]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(x1,y1,marker='o',facecolors='none', edgecolors='none')
    for i, label in enumerate(df.index):
        ax.annotate(label, (x1.iloc[i], y1.iloc[i]), size=16, color='gray')
    ax.plot(np.unique(x1),
            np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)),
            color='gray')
    ax.set_xlabel(xlab,size=20)
    ax.set_ylabel(ylab,size=20)
    # Super impose with a new set
    ax.scatter(x2,y2,marker='o',facecolors='none', edgecolors='none')
    for i, label in enumerate(df.index):
        ax.annotate(label, (x2.iloc[i], y2.iloc[i]), size=16, color='blue')
    ax.plot(np.unique(x2),
            np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)),
            color='blue')
    ax.set_xlabel(xlab,size=20)
    ax.set_ylabel(ylab,size=20)    
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    return fig, ax

# ---------- x. # Plot for GEM note
def plot_gem(cset=['US']): 
    # Graph 1: Baseline projections
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), constrained_layout=True)        
    transpa = 0
    data1 = {}
    df_yratio = pd.read_csv(f'../output/growth-mob.csv', index_col=0)
    for c in cset:
        tmp = pickle.load(open(f'../output/{out_load_folder}/{c}_baseline.pkl','rb'))
        df = tmp.df3
        yearend = df.index.get_loc('2020-12-31')+1
        gdp_2020 = df_yratio.loc[c]['Growth']
        mob_2021 = df['mob_fc'].iloc[yearend:].mean() # Average mobility for 2021
        gdp_2021 = 100*mob_2021*df_yratio.loc[c]['ym_ratio']
        data1[c] = [gdp_2020, gdp_2021]
        if c == 'US':
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='red', label='US')
            ax[0,1].plot(df.index, 100*df['DD_N'], color='red', label='US')
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='red', label='US')
        elif c =='DE':
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='blue',label='DE')
            ax[0,1].plot(df.index, 100*df['DD_N'], color='blue', label='DE')
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='blue', label='DE')
        elif c =='GB':
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='darkgreen', label='GB')
            ax[0,1].plot(df.index, 100*df['DD_N'], color='darkgreen', label='GB')
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='darkgreen', label='GB')
        elif c =='BR':
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='orange', label='BR')
            ax[0,1].plot(df.index, 100*df['DD_N'], color='orange', label='BR')
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='orange', label='BR')
        elif c =='IN':
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='lightskyblue', label='IN')
            ax[0,1].plot(df.index, 100*df['DD_N'], color='lightskyblue', label='IN')
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='lightskyblue', label='IN')
        else:
            ax[0,0].plot(df.index, 100*df['DT_N'].diff(), color='lightgray', alpha=0.6)
            ax[0,1].plot(df.index, 100*df['DD_N'], color='lightgray', alpha=0.6)
            ax[1,0].plot(df.index, 100*df['mob_fc'], color='lightgray', alpha=0.6) 
    ax[0,0].legend(loc=0,framealpha=transpa ,fontsize='x-large')
    ax[0,1].legend(loc=0,framealpha=transpa ,fontsize='x-large')
    ax[1,0].legend(loc=0,framealpha=transpa ,fontsize='x-large')
    ax[0,0].axvline(df.index[tmp.T], linewidth = 1, color='black', linestyle='-')
    ax[0,1].axvline(df.index[tmp.T], linewidth = 1, color='black', linestyle='-')
    ax[1,0].axvline(df.index[tmp.T], linewidth = 1, color='black', linestyle='-')
    plt.setp(ax[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    # Last panel, scatter plot
    df1 =  pd.DataFrame.from_dict(data1, orient='index', columns=['GDP 2020','GDP 2021']) 
    ax[1,1].scatter(df1['GDP 2020'],df1['GDP 2021'],marker='o',facecolors='none', edgecolors='none')
    for i, label in enumerate(df1.index):
        ax[1,1].annotate(label, (df1['GDP 2020'].iloc[i], df1['GDP 2021'].iloc[i]), size=16)
    ax[1,1].axline([0, 0], [1, 1])
    ax[1,1].set_xlabel(xlab,size=20)
    ax[1,1].set_ylabel(ylab,size=20)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    Path(f'../pics/GEM_Mar2021').mkdir(exist_ok=True)
    fig.savefig(f'../pics/GEM_Mar2021/fig1_11Feb.png')      
    # Graph 2: Downside risks
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), constrained_layout=True)        
    # transpa = 0
    # for c in cset:
    #     tmp = pickle.load(open(f'../output/{out_load_folder}/{c}_baseline.pkl','rb'))
    #     df = tmp.df3
    #     yearend = df.index.get_loc('2020-12-31')+1        


# Plot 2 countries' baseline versus one scenario
def plot_scenarios(cset = ['US','DE'],scenario='vacworse'):
    fig, ax = plt.subplots(nrows=2, ncols=len(cset), figsize=(5*len(cset),8), constrained_layout=True)
    transpa = 0.0
    col = 0
    for c in cset:
        tmp = pickle.load(open(f'../output/Agustin presentation/{c}_baseline.pkl','rb'))
        tmp_s = pickle.load(open(f'../output/Agustin presentation/{c}_{scenario}.pkl','rb'))
        df = tmp.df3
        df_s = tmp_s.df3
        ax[0,col].plot(df.index, 100*df_s['S']/tmp.N, label='$S_t$',color='red')
        ax[0,col].plot(df.index, 100*df_s['V']/tmp.N, label='$V_t$',color='red',linestyle='--')
        ax[0,col].plot(df.index, 100*df['S']/tmp.N, label='baseline',color='gray')
        ax[0,col].plot(df.index, 100*df['V']/tmp.N, color='gray',linestyle='--')
        ax[0,col].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[0,col].set_title(f'{c}: susceptible & vaccinated',fontsize='x-large')
        ax[0,col].legend(loc='best',framealpha=transpa ,fontsize='x-large')
        ax[0,col].set(ylabel='% of population')
    
        ax[1,col].plot(df.index, 100*df_s['mob_fc'], label='Mobility', color='red')    
        ax[1,col].plot(df.index, 100*df['mob_fc'], label='baseline', color='gray')
        ax[1,col].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
        ax[1,col].legend(loc=0,framealpha=transpa ,fontsize='x-large')
        ax[1,col].set_title(f'{c}: mobility',fontsize='x-large')
        ax[1,col].set(ylabel='% deviations from norm')

        plt.setp(ax[0,col].get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax[1,col].get_xticklabels(), rotation=30, horizontalalignment='right')
        col =+ 1
    Path(f'../pics/Agustin').mkdir(exist_ok=True)
    fig.savefig(f'../pics/Agustin/fig-{scenario}-{cset}.png')

def all_output(cset=['US','DE']):
    data_col = ['Mob 2021','Mob fc',
                        'GDP 2021','GDP fc',
                        'dDeath 2021','dDeath fc',
                        'dD/mn 2021','dD/mn fc',
                        'Mob 2021 3rdwave', 'Mob fc 3rdwave',
                        'GDP 2021 3rdwave', 'GDP fc 3rdwave',
                        'dDeath 2021 3rdwave', 'dDeath fc 3rdwave',
                        'dD/mn 2021 3rdwave', 'dD/mn fc 3rdwave',
                        'Mob 2021 vacworse', 'Mob fc vacworse',
                        'GDP 2021 vacworse', 'GDP fc vacworse',
                        'dDeath 2021 vacworse', 'dDeath fc vacworse',
                        'dD/mn 2021 vacworse', 'dD/mn fc vacworse',
                        'Mob 2021 reinfect', 'Mob fc reinfect',
                        'GDP 2021 reinfect', 'GDP fc reinfect',
                        'dDeath 2021 reinfect', 'dDeath fc reinfect',
                        'dD/mn 2021 reinfect', 'dD/mn fc reinfect',
                        'Mob 2021 better', 'Mob fc better',
                        'GDP 2021 better', 'GDP fc better',
                        'dDeath 2021 better', 'dDeath fc better',
                        'dD/mn 2021 better', 'dD/mn fc better',
                        ]
    data = {}
    df_yratio = pd.read_csv(f'../output/growth-mob.csv', index_col=0)
    for c in cset:
        # tmp = pickle.load(open(f'../output/Agustin presentation/{c}_baseline.pkl','rb'))
        # tmp2 = pickle.load(open(f'../output/Agustin presentation/{c}_vacworse.pkl','rb'))
        # tmp3 = pickle.load(open(f'../output/Agustin presentation/{c}_reinfect.pkl','rb'))
        tmp = pickle.load(open(f'../output/{out_load_folder}/{c}_baseline.pkl','rb'))
        tmp1 = pickle.load(open(f'../output/{out_load_folder}/{c}_shock.pkl','rb'))
        tmp2 = pickle.load(open(f'../output/{out_load_folder}/{c}_vacworse.pkl','rb'))
        tmp3 = pickle.load(open(f'../output/{out_load_folder}/{c}_reinfect.pkl','rb'))
        tmp4 = pickle.load(open(f'../output/{out_load_folder}/{c}_better.pkl','rb'))
        cnum = tmp.df3.index.get_loc('2020-12-31')+1
        d = tmp.df3['total_cases'].last_valid_index()
        dnum = tmp.df3.index.get_loc(d)+1
        
        mob_2021 = tmp.df3['mob_fc'].iloc[cnum:].mean() # Average mobility for 2021
        mob_fc = tmp.df3['mob_fc'].iloc[dnum:].mean() # Average mobility from current date till year end
        GDP_2021 = 100*mob_2021*df_yratio.loc[c]['ym_ratio']
        GDP_fc = 100*mob_fc*df_yratio.loc[c]['ym_ratio']
        dD_2021 = tmp.df3['DD'][-1] - tmp.df3['DD'][cnum]    
        dD_fc = tmp.df3['DD'][-1] - tmp.df3['DD'][dnum]
        dD_mn_2021 = 1000000*dD_2021/tmp.N
        dD_mn_fc = 1000000*dD_fc/tmp.N
        
        mob_2021_shock = tmp1.df3['mob_fc'].iloc[cnum:].mean() # Average mobility for 2021
        mob_fc_shock = tmp1.df3['mob_fc'].iloc[dnum:].mean() # Average mobility from current date till year end
        GDP_2021_shock = 100*mob_2021_shock*df_yratio.loc[c]['ym_ratio']
        GDP_fc_shock = 100*mob_fc_shock*df_yratio.loc[c]['ym_ratio']
        dD_2021_shock = tmp1.df3['DD'][-1] - tmp1.df3['DD'][cnum]    
        dD_fc_shock = tmp1.df3['DD'][-1] - tmp1.df3['DD'][dnum]
        dD_mn_2021_shock = 1000000*dD_2021_shock/tmp.N
        dD_mn_fc_shock = 1000000*dD_fc_shock/tmp.N
        
        mob_2021_vacworse = tmp2.df3['mob_fc'].iloc[cnum:].mean() # Average mobility for 2021
        mob_fc_vacworse = tmp2.df3['mob_fc'].iloc[dnum:].mean() # Average mobility from current date till year end
        GDP_2021_vacworse = 100*mob_2021_vacworse*df_yratio.loc[c]['ym_ratio']
        GDP_fc_vacworse = 100*mob_fc_vacworse*df_yratio.loc[c]['ym_ratio']
        dD_2021_vacworse = tmp2.df3['DD'][-1] - tmp2.df3['DD'][cnum]    
        dD_fc_vacworse = tmp2.df3['DD'][-1] - tmp2.df3['DD'][dnum]
        dD_mn_2021_vacworse = 1000000*dD_2021_vacworse/tmp.N
        dD_mn_fc_vacworse = 1000000*dD_fc_vacworse/tmp.N

        mob_2021_reinfect = tmp3.df3['mob_fc'].iloc[cnum:].mean() # Average mobility for 2021
        mob_fc_reinfect = tmp3.df3['mob_fc'].iloc[dnum:].mean() # Average mobility from current date till year end
        GDP_2021_reinfect = 100*mob_2021_reinfect*df_yratio.loc[c]['ym_ratio']
        GDP_fc_reinfect = 100*mob_fc_reinfect*df_yratio.loc[c]['ym_ratio']
        dD_2021_reinfect = tmp3.df3['DD'][-1] - tmp3.df3['DD'][cnum]    
        dD_fc_reinfect = tmp3.df3['DD'][-1] - tmp3.df3['DD'][dnum]
        dD_mn_2021_reinfect = 1000000*dD_2021_reinfect/tmp.N
        dD_mn_fc_reinfect = 1000000*dD_fc_reinfect/tmp.N  

        mob_2021_better = tmp4.df3['mob_fc'].iloc[cnum:].mean() # Average mobility for 2021
        mob_fc_better = tmp4.df3['mob_fc'].iloc[dnum:].mean() # Average mobility from current date till year end
        GDP_2021_better = 100*mob_2021_better*df_yratio.loc[c]['ym_ratio']
        GDP_fc_better = 100*mob_fc_better*df_yratio.loc[c]['ym_ratio']
        dD_2021_better = tmp4.df3['DD'][-1] - tmp4.df3['DD'][cnum]    
        dD_fc_better = tmp4.df3['DD'][-1] - tmp4.df3['DD'][dnum]
        dD_mn_2021_better = 1000000*dD_2021_better/tmp.N
        dD_mn_fc_better = 1000000*dD_fc_better/tmp.N  
        
        data[c] = [mob_2021,mob_fc,
                   GDP_2021,GDP_fc,
                   dD_2021,dD_fc,
                   dD_mn_2021,dD_mn_fc,
                   mob_2021_shock, mob_fc_shock,
                   GDP_2021_shock, GDP_fc_shock,
                   dD_2021_shock, dD_fc_shock,
                   dD_mn_2021_shock, dD_mn_fc_shock,
                   mob_2021_vacworse, mob_fc_vacworse,
                   GDP_2021_vacworse, GDP_fc_vacworse,
                   dD_2021_vacworse, dD_fc_vacworse,
                   dD_mn_2021_vacworse, dD_mn_fc_vacworse,
                   mob_2021_reinfect, mob_fc_reinfect,
                   GDP_2021_reinfect, GDP_fc_reinfect,
                   dD_2021_reinfect, dD_fc_reinfect,
                   dD_mn_2021_reinfect, dD_mn_fc_reinfect,
                   mob_2021_better, mob_fc_better,
                   GDP_2021_better, GDP_fc_better,
                   dD_2021_better, dD_fc_better,
                   dD_mn_2021_better, dD_mn_fc_better,
                   ]
    df_out =  pd.DataFrame.from_dict(data, orient='index', columns=data_col) 
    # fig, ax = scatter1('GDP 2021','dD/mn 2021','GDP loss (2021)','Deaths per million (2021)',df_out)
    # ax.set_title('Baseline', size=20, fontweight="bold")
    # fig.savefig('../pics/GEM_Mar2021/fig_GDP_deaths_baseline.png', dpi=300)
    # fig3, ax = scatter2('GDP 2021','dD/mn 2021','GDP 2021 vacworse', 'dD/mn 2021 vacworse', 'GDP loss (2021)', 'Deaths per million (2021)', df_out)
    # ax.set_title('Limited vaccines', size=20,fontweight="bold")
    # fig3.savefig('../pics/GEM_Mar2021/fig_GDP_deaths_vacworse.png', dpi=300)
    # fig4, ax = scatter2('GDP 2021','dD/mn 2021','GDP 2021 reinfect', 'dD/mn 2021 reinfect', 'GDP loss (2021)', 'Deaths per million (2021)', df_out)
    # ax.set_title('Virus reinfection', size=20,fontweight="bold")
    # fig4.savefig('../pics/GEM_Mar2021/fig_GDP_deaths_reinfect.png', dpi=300)
    # fig5, ax = scatter2('GDP 2021','dD/mn 2021','GDP 2021 better', 'dD/mn 2021 better', 'GDP loss (2021)', 'Deaths per million (2021)', df_out)
    # ax.set_title('Better', size=20,fontweight="bold")
    # fig5.savefig('../pics/GEM_Mar2021/fig_GDP_deaths_better.png', dpi=300)
    
    name = f'../output/{out_save_folder}/all_output.pkl'
    pickle.dump(df_out,open(name,'wb'))
    return df_out

# Compare 2 scenarios, showing 4 key charts
def plot_2cases(c,df,df2,tmp,title,filename,saveplot=False):
    # Compute differences between 2 cases (lives saved at end points, and average mobility differences)
    d_death = int(round(df2['fitted_deaths'][-1]-df['fitted_deaths'][-1],0))
    d_m = round((df2['fitted_m']['2021-01-01':'2021-12-31'] - df['fitted_m']['2021-01-01':'2021-12-31']).mean(),3)
    # Draw figure
    fig3, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), constrained_layout=True)
    ax[0,0].plot(df.index, 100*df['fitted_cases']/tmp.N, linewidth = 2, label='Cases', color='red')
    ax[0,0].plot(df2.index, 100*df2['fitted_cases']/tmp.N, linewidth = 2, label='Cases alt', color='red', linestyle=':')
    ax[0,0].plot(df.index, 100*df['fitted_I']/tmp.N, linewidth = 2, label='Infected', color='green')
    ax[0,0].plot(df2.index, 100*df2['fitted_I']/tmp.N, linewidth = 2, label='Infected alt', color='green',linestyle=':')
    ax[0,0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[0,0].legend(loc='center right',fontsize='x-large',fancybox=True, framealpha=0.5)
    ax[0,0].set_title('Cases',fontsize='x-large')
    ax[0,1].plot(df.index, 100*df['fitted_deaths']/tmp.N, linewidth = 2, label='Deaths', color='red')
    ax[0,1].plot(df2.index, 100*df2['fitted_deaths']/tmp.N, linewidth = 2, label='Deaths alt', color='red', linestyle=':')
    ax[0,1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[0,1].legend(loc='lower right',fontsize='x-large',fancybox=True, framealpha=0.5)
    ax[0,1].set_title('Deaths',fontsize='x-large')
    ax[0,1].annotate(f'$\Delta$Death={d_death}', xy=(0.1,0.9), xycoords='axes fraction')
    ax[1,0].plot(df.index, 100*df['fitted_S']/tmp.N, linewidth = 2, label='S ', color='red')
    ax[1,0].plot(df2.index, 100*df2['fitted_S']/tmp.N, linewidth = 2, label='S alt', color='red',linestyle=':')
    ax[1,0].plot(df.index, 100*df['fitted_V']/tmp.N, linewidth = 2, label='V ', color='green')
    ax[1,0].plot(df2.index, 100*df2['fitted_V']/tmp.N, linewidth = 2, label='V alt', color='green',linestyle=':')
    ax[1,0].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[1,0].legend(loc='upper right',fontsize='x-large',fancybox=True, framealpha=0.5)
    ax[1,0].set_title('Susceptible & Vaccinated',fontsize='x-large')
    ax[1,1].plot(df.index, 100*df['fitted_m'], linewidth = 2, label='Mobility', color='red')
    ax[1,1].plot(df2.index, 100*df2['fitted_m'], linewidth = 2, label='Mobility alt', color='red', linestyle=':')
    ax[1,1].axvline(df.index[tmp.T], linewidth = 2, color='gray', linestyle=':')
    ax[1,1].legend(loc='lower right',fontsize='x-large',fancybox=True, framealpha=0.5)
    ax[1,1].set_title('Mobility',fontsize='x-large')
    ax[1,1].annotate(f'$\Delta$m={d_m}', xy=(0.1,0.9), xycoords='axes fraction')
    plt.setp(ax[0,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    ax[0,0].set_ylabel('Percent of population')
    ax[0,1].set_ylabel('Percent of population')
    ax[1,0].set_ylabel('Percent of population')
    ax[1,1].set_ylabel('Percent deviations from norm')
    fig3.suptitle(f'{c}: {title}',fontsize='x-large')
    if saveplot:
        Path(f'../pics/fig_{date.today()}').mkdir(exist_ok=True)
        fig3.savefig(f'../pics/fig_{date.today()}/fig-{tmp.iso2}-{filename}.png')

    
# ------------------------------
# x. Diagnostic/Inspect functions

# Plot m_t and gamma_t (data)
def plot_m_gamma(cset=['US','DE']):
    no_fig = len(cset)
    nrows_ = round(no_fig/2)
    ncols_ = 2
    fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(14,6*nrows_))
    i = 0
    j = 0 
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        tmp.gamma_t_compute()
        ax[i,j].plot(tmp.df2['gamma_sm'], label = 'gamma_sm', color='black')
        ax2 = ax[i,j].twinx()
        ax2.plot(tmp.df2['google_smooth'], label='mobility',color='blue')
        lines, labels = ax[i,j].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize='x-large')
        plt.setp(ax[i,j].get_xticklabels(), rotation=30, horizontalalignment='right')
        ax[i,j].set_title(f'{c}')
        if j == 0:
            j +=1
        else:
            j = 0
            i +=1

# Plot realised mobility against model-implied I/N
def policy_check():
    cset = ['US','DE','FR']
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,6), constrained_layout=True)  
    n = 0
    dict = {}
    for c in cset:
        tmp = solveCovid(c)
        tmp.prelim()
        param = mod.get_params('round2')['all']
        fig_tmp, df = tmp.predict(params=param,plot=False,saveplot=False)
        dfa = df[['google_smooth','fitted_I']].dropna()
        dfa.loc[:,'fitted_I'] = dfa['fitted_I']/tmp.N
        dfa = dfa.iloc[-120:-1]
        ax[n].plot(dfa.index, dfa['google_smooth'],label='mobility')
        ax2 = ax[n].twinx()
        ax2.plot(dfa.index, dfa['fitted_I'],label='infected',color='g')
        ax[n].set_title(c,fontsize='x-large')
        n +=1
        # Regressions on more recent samples
        X = sm.add_constant(dfa['fitted_I'])
        y = dfa['google_smooth']
        model = sm.OLS(y,X).fit()
        dict[c]=model.summary()
    return dict
        
# Print estimated parameters
def printparams(param_folder:str, cset:list):
    # alpha, r_dth, p_dth0, b_dth, c_dth, k1, k2, 
    # beta0, beta1, a, b, c, a2, b2, c2
    dict = {'Parameters': ['alpha','r_dth','p_dth0','b_dth','c_dth','k1','k2',
                            'beta0','beta','a','b','c',
                            'a2','b2','c2']} 
    for c in cset:
        # p1 = pickle.load(open(f'../params/{param_folder}/estimates-{c}-round1.pkl','rb'))
        # p2 = pickle.load(open(f'../params/{param_folder}/estimates-{c}-round2.pkl','rb'))
        # dict[c] = tuple(p1)+tuple(p2)
        p = pickle.load(open(f'../params/{param_folder}/estimates-{c}-joint.pkl','rb'))
        dict[c] = tuple(p)
    df = pd.DataFrame(dict) 
    pd.options.display.float_format = '{:,.2f}'.format
    display(df)
    df.to_csv(f'../params/{param_folder}/summary.csv',float_format='%.2f',index=False)






