# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:49:50 2021

@author: Phurichai Rungcharoenkitkul

Update the excel file from Bloomberg
https://www.bloomberg.com/graphics/covid-vaccine-tracker-global-distribution/
https://www.bloomberg.com/graphics/covid-vaccine-tracker-global-distribution/contracts-purchasing-agreements.html
https://www.nature.com/articles/d41586-020-03370-6

Note:
    - As of 27 Jan 2021, pre-orders amount to about 4 billion people 
    - From Nature article, the top 3 vaccine producers can supply for 2.6 bn people in 2021 
    - 2021 Supply = 65% of demand: Need Q2 2022 to get all vaccines

"""
import country_converter as coco 
from country import country
import pandas as pd
import pickle

data_sheet = '9 Mar'

df = pd.read_excel("../data/vaccine_contracts.xlsx",sheet_name=data_sheet,
                    engine='openpyxl',)

#out = coco.convert(names=df['Country/Region'].values,to='iso2')

for i in range(len(df)):
    df['iso2'].iloc[i] = coco.convert(names=df['cname'].iloc[i], to='iso2')

df.index = df['iso2']    
df.to_excel(f'../data/vaccine_out_{data_sheet}.xlsx')

name = f'../data/vaccine_out_{data_sheet}.pkl'
pickle.dump(df,open(name,'wb'))
