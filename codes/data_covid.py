# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:55:11 2020

@author: Phurichai Rungcharoenkitkul

Data sources: Our World in Data, Google 
https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv

"""
import pandas as pd
import numpy as np
import urllib.request 
from datetime import date
import time
import pickle
import country_converter as coco 
from country import country
import pdb

#--------------------
# Define country list
c_all2 = country('all2','ISO2')
c_all2b = country('all2','ISO3')
# --------------------
# Download data
tic = time.perf_counter()
urllib.request.urlretrieve("https://covid.ourworldindata.org/data/owid-covid-data.xlsx", "../data/ourworld.xlsx") 
urllib.request.urlretrieve("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv","../data/google.csv")
df0 = pd.read_excel("../data/ourworld.xlsx",sheet_name="Sheet1",
                    engine='openpyxl',)
df1 = pd.read_csv("../data/google.csv",
                  dtype={'sub_region_1': 'str',
                         'sub_region_2': 'str',
                         'metro_area': 'str',
                         'iso_3166_2_code': 'str'})
# --------------------
# Data cleaning
df0 = df0[df0['iso_code'].isin(c_all2b)]
df1 = df1[df1['country_region_code'].isin(c_all2)]

df1=df1[df1['sub_region_1'].isna()]
df1=df1[df1['metro_area'].isna()]

df1=df1.rename(columns={'retail_and_recreation_percent_change_from_baseline': 'google_retail_and_recreation',
                    'grocery_and_pharmacy_percent_change_from_baseline': 'google_grocery_and_pharmacy',
                    'parks_percent_change_from_baseline': 'google_parks',
                    'transit_stations_percent_change_from_baseline': 'google_transit',
                    'workplaces_percent_change_from_baseline': 'google_workplaces',
                    'residential_percent_change_from_baseline': 'google_residential'})

df0['iso2']=coco.convert(names=df0['iso_code'].values.tolist(),to='ISO2') # could be more efficient here
df = pd.merge(df0,df1, how = 'left', left_on=['iso2','date'], right_on=['country_region_code','date'])
df = df.drop(columns=['country_region_code','country_region',
                      'sub_region_1','sub_region_2','metro_area','iso_3166_2_code', 'census_fips_code'])

#df['google_mobility'] = df[[col for col in df if col.startswith('google')]].sum(axis=1)
df['google_mobility'] = (df['google_retail_and_recreation']+df['google_transit']+df['google_workplaces'])/3

df['new_cases'] = df['total_cases'].diff()
df['new_deaths'] = df['total_deaths'].diff()
# ------------------
# Pre-processing
# df = df[~df['iso2'].isin(['HK','CN','CY'])]
df1 = df.pivot_table(values=['total_cases','total_deaths',
                             'new_cases','new_deaths',
                             'total_cases_per_million','total_deaths_per_million',
                             'population',
                             'google_mobility',
                             'stringency_index',
                             'hospital_beds_per_thousand',
                             'median_age','aged_65_older','aged_70_older',
                             'cardiovasc_death_rate','diabetes_prevalence',
                             'gdp_per_capita',
                             'reproduction_rate',
                             'icu_patients','hosp_patients',
                             'new_tests','tests_per_case',
                             'total_vaccinations_per_hundred',
                             'people_vaccinated_per_hundred',
                             'people_fully_vaccinated_per_hundred'],
                     index='date', 
                     columns=['iso2'],
                     dropna=False)
df1.index = pd.to_datetime(df1.index)
df_sm = df1['google_mobility'].rolling(7).mean().bfill().ffill()/100
df_sm2 = pd.concat({'google_smooth': df_sm},axis=1)
df2 = df1.merge(df_sm2, how='left', left_index=True, right_index=True)

df2 = df2.rename(columns={'total_vaccinations_per_hundred': 'vac_total',
                    'people_vaccinated_per_hundred': 'vac_partial',
                    'people_fully_vaccinated_per_hundred': 'vac_fully'})
# ------------------
# Save data into file
name = "../data/data_daily_"+str(date.today()) +'.pkl'
pickle.dump(df2,open(name,'wb'))


toc = time.perf_counter()
print(f'Data created in {toc - tic:0.4f} seconds')
