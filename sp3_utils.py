#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:10:14 2022
  sp3_utils.py:  parsing NGA SP3 files
@author: mcarroll
"""

import re
import pandas as pd
import numpy as np
p = re.compile(r'\s+')


def parse_timestamp(line):
#  Could also use:  datetime.fromisoformat('2011-11-04T00:05:23')
    fields = p.split(line)
    ts_str = f'{fields[1]}-{fields[2]}-{fields[3]}T{fields[4]}:{fields[5]}:{fields[6]}{fields[7]}'
    ts = pd.to_datetime(ts_str)
    return ts_str,ts

def gen_csvrec(timestamp_str,p_rec,v_rec):
    pfields = p.split(p_rec)
    vfields = p.split(v_rec)
    if pfields[1]==vfields[1]:
        csvrec = f'{timestamp_str},{pfields[1]},{pfields[2]},{pfields[3]},{pfields[4]},{pfields[5]}'
        csvrec = f'{csvrec},{vfields[2]},{vfields[3]},{vfields[4]},{vfields[5]}'
        result = True
    else:
        csvrec = 'SV mismatch'
        result = False

    return result, csvrec
        
def get_df(filename):
    # df = pd.read_csv(filename, index_col=['recnum','timestamp'],\
    #                  parse_dates=['recnum','timestamp'],\
    #                      infer_datetime_format=True,\
    #                          keep_date_col=True)
    df = pd.read_csv(filename)
    df['r']= np.sqrt(df['x']**2+df['y']**2+df['z']**2)
    return df
#    df['timestamp']=pd.to_datetime(df['timestamp'])
#    df.set_index('timestamp', inplace=True)

def filter_df_for_prn(df, prn):
    prn_filter = (df['PRN']==prn)
    prn_df = df[prn_filter]
    return prn_df

def get_vec_diff(df1,df2):
    x1 = np.array(df1['x'])
    y1 = np.array(df1['y'])
    z1 = np.array(df1['z'])
    x2 = np.array(df2['x'])
    y2 = np.array(df2['y'])
    z2 = np.array(df2['z'])
    dx = x1-x2
    dy = y1-y2
    dz = z1-z2
    diff = np.array([dx,dy,dz]).reshape((len(x1),3))
    return diff
    