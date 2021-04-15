# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:39:18 2021

@author: Phurichai Rungcharoenkitkul
"""

from datetime import datetime
from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------
# Choose data and estimates cohort
# ------------------------------

chosen_date = str(date.today()) 
#chosen_date = '2021-03-11' # Choose a fixed date to simulate using already downloaded data (eg older cohorts)
#chosen_date = '2021-04-07'
vac_date = '9 Mar'

df1 = pd.read_pickle(f'../data/data_daily_{chosen_date}.pkl')
param_save_folder = f'param_{chosen_date}'
param_load_folder = f'param_{chosen_date}'
out_save_folder = f'output_{chosen_date}'
out_load_folder = f'output_{chosen_date}'

Path(f'../params/{param_save_folder}').mkdir(exist_ok=True)
Path(f'../params/{param_load_folder}').mkdir(exist_ok=True)
Path(f'../params/param_fixed').mkdir(exist_ok=True)
Path(f'../output/{out_save_folder}').mkdir(exist_ok=True)
Path(f'../output/{out_load_folder}').mkdir(exist_ok=True)

f_table = pd.read_pickle('../data/age_fatality.pkl')

df_vac = pd.read_pickle(f'../data/vaccine_out_{vac_date}.pkl')

# ----- Sub-samples for multi-round estimation
T1_date = datetime(2020,12,31) # End of round-1 sample (before 2nd wave)

# -------------------------------------
# Basic settings
# -------------------------------------
virus_thres = 100 # Epidemic starting point
default_maxT = datetime(2021, 12, 31)  # Forecast end point
max_iter = 5000  # Maximum number of iterations for the algorithm

# ------------------------------------
# Epidemiological assumptions
# ------------------------------------
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated
p_v = 0.25  # Percentage of ventilated
p_d = 0.2  # Percentage of infection cases detected.
p_h = 0.15  # Percentage of detected cases hospitalized

# Derived fixed coefficients
r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
r_d = np.log(2) / DetectD  # Rate of detection
r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
r_dth = 0.5


# ---------------------------------
# Lockdown assumption
# ---------------------------------
max_lockdown = -0.8 # Maximum lockdown 

# -----------------------------------
# Initial guesses for estimation
# -----------------------------------
# alpha, r_dth, p_dth0, b_dth, c_dth, k1, k2, 
# beta0, beta1, a, b, c, a2, b2, c2

a_0 = 0.3
b_0 = 100
c_0 = 15
beta1_0 = 2.5
beta0_0 = r_d * 6 # when m=0, gamma_t = gamma_0

#------- Initial guesses & bounds 
default_init_single = [beta0_0, beta1_0]
default_bounds_single = [
    (0, 15), (1,15)
    ]


# -------- Verify that the latest parameters load
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Loading parameters as of =", current_time)