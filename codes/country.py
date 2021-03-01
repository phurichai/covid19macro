# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:43:23 2020

@author: Phurichai Rungcharoenkitkul
"""
import country_converter as coco 

## To check the full list
# cc = coco.CountryConverter()
# cc.valid_class


def country(cset,form):
    adv = ['US','XM','JP','GB','AU','CA','DK','NZ','NO','SE','CH']
    asi = ['CN','TW','HK','IN','ID','KR','MY','PH','SG','TH']
    lat = ['AR','BR','CL','CO','MX','PE']
    cee = ['CZ','HU','IL','PL','RU','TR']
    oth = ['ZA','SA']
    c_all = adv+asi+lat+cee+oth
    EA = ['AT','BE','CY','EE','FI','FR','DE','GR','IE','IT','LV','LT','LU','MT','NL','PT','SK','SI','ES']
    if cset == 'adv1':
        c_set = adv
    elif cset == 'adv2':
        c_set = list(set(adv) - set(['XM'])) + EA
    elif cset == 'asi':
        c_set = asi
    elif cset == 'lat':
        c_set = lat
    elif cset == 'all1':
        c_set = c_all
    elif cset == 'all2':
        c_set = list(set(c_all) - set(['XM'])) + EA
    elif cset == 'special1': # For Covid update
        c_set = list(set(c_all+ EA) - set(['XM','HK','CN','CY','MT','TW'])) 
    elif cset == 'per':
        c_set = list(set(c_all) - set(['XM','CZ','HU','SA'])) + ['FR','DE','IT','ES','NL']
        
    if form == 'ISO2':
        output = c_set
    elif form == 'ISO3':
        output = coco.convert(names=c_set,to='ISO3')
    elif form == 'UNcode':
        output = coco.convert(names=c_set,to='UNcode')
    elif form == 'UN':
        output = coco.convert(names=c_set,to='UN')
    elif form == 'name_official':
        output = coco.convert(names=c_set,to='name_official')
    
    return output

