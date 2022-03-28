#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:40:28 2022
    pspredict_clock.py
    Using pysindy to derive a model of GPS ephemeris dynamics.
    Goal is to generate a model that predicts more accurately than
    the NGA 9-day orbit ephemeris.  (See compare_predicts.py for an assessment of the
    9-day accuracy based on after-the-face NGA PEs.)
    Training data is GPS pcenter of mass recise ephemeris data downloaded from NGA public site:
        https://earth-info.nga.mil/index.php?dir=gnss&action=gnss
    9-day orbit predictions can be downloaded from there as well.
    NGA PE and Predict data have been converted to pandas-friendly csv
    files and our posted on this github site:
        https://github.com/musicarroll/gps
    Typical runtime arguments for this script are:
        csv/ngape.csv csv/predicts.csv --prn 1 --num_train_days 14 
            --train_ts 2022-01-31T23:55:00.0 --test_ts 2022-02-09T23:55:00.0
        Note: Currently there are only a maximum of 62 days of training data available
        on the github site.  More can be downloaded from NGA and preprocessed with
        readeph.py if desired.
    Pysindy citation:
        Brian M. de Silva, Kathleen Champion, Markus Quade, Jean-Christophe Loiseau, 
        J. Nathan Kutz, and Steven L. Brunton., (2020). 
        PySINDy: A Python package for the sparse identification of nonlinear dynamical 
        systems from data. Journal of Open Source Software, 5(49), 2104, 
        https://doi.org/10.21105/joss.02104
        Kaptanoglu et al., (2022). PySINDy: A comprehensive Python package for robust 
        sparse system identification. Journal of Open Source Software, 7(69), 3994, 
        https://doi.org/10.21105/joss.03994
    Status:
        2022-03-05:  Added clock and clock rate variables t and v_t to the model 
        originally coded up in pspredict.py.  Results are now MSE 35 m, whereas 
        before they were on the order of 45 (after some tuning advice from 
        Alan Kaptanoglu.)
@author: mcarroll
"""

import pysindy as ps
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sp3_utils as su
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('infile', help='Input file:  Required!!!')
#parser.add_argument('predicts', help='Predicts input file. Required!!!')
parser.add_argument('--prn', help='Specific PRN to look at.')
parser.add_argument('--train_ts', help='Last timestamp of training data.')
parser.add_argument('--test_ts', help='Last timestamp of test data.')
parser.add_argument('--num_train_days', help = 'Number of days prior to --train_ts to include')
args = parser.parse_args()

km_to_cm = 100*1000
microsec_to_sec = 1e-6
speed_of_light_m_per_sec = 299_792_458
m_to_km = 1e-3
dm_to_m = 1e-1
dm_per_sec_to_km_per_sec = 1/10000

if args.infile is not None:
    df = su.get_df(args.infile)    
# if args.predicts is not None:
#     predicts = su.get_df(args.predicts)

df['timestamp']=pd.to_datetime(df['timestamp'])
df[['v_x','v_y','v_z']]= df[['v_x','v_y','v_z']]*dm_per_sec_to_km_per_sec
df['v_t']= df['v_t']* speed_of_light_m_per_sec * microsec_to_sec * m_to_km # km/s 
df['t'] = df['t'] * speed_of_light_m_per_sec * microsec_to_sec * m_to_km # km 

# ngapts = np.array(predicts['timestamp'])
# filt = df['timestamp'].isin(ngapts)
# df_truth = df[filt]
if args.prn is not None:
    prnfilt = df['PRN']==int(args.prn)
    prn_df = df[prnfilt]
    train_ts = pd.to_datetime(args.train_ts)
    if args.num_train_days is not None:
        num_train_days = int(args.num_train_days)
        train_start_ts = train_ts - pd.Timedelta(num_train_days,'days')
        train_filt = (prn_df['timestamp']>train_start_ts) & (prn_df['timestamp']<train_ts)
    else:
        train_filt = (prn_df['timestamp']<train_ts)
    train_df = prn_df[train_filt]

    test_ts = pd.to_datetime(args.test_ts)
    test_filt = (prn_df['timestamp']>train_ts) & (prn_df['timestamp']<=test_ts)
    test_df  = prn_df[test_filt]
    
# Set up time array:
#    dt=300  # = 5 min sample rate in seconds
    t=np.array(60*np.arange(0,len(prn_df)*5,5))
    t_train = t[:len(train_df)]
    t_test  = t[len(train_df):len(train_df)+len(test_df)]

# Derivatives:  Pre-computed, default or from NGA velocity estimates:
#    x_dot_precomputed = ps.FiniteDifference()._differentiate(trainer.values, t_train)


    vel_train = train_df[['v_x','v_y','v_z','v_t']]
    
    vel_test = test_df[['v_x','v_y','v_z','v_t']] 
 
# Function Libraries:
    fourier_library = ps.FourierLibrary(n_frequencies=2)
#    identity_library = ps.IdentityLibrary()
    poly_library = ps.PolynomialLibrary(include_bias=True)
    my_library = poly_library+fourier_library
#    my_library = fourier_library  # Worse performance MSE of 3 km!!

# Sparse Optimization methods:
#    optimizer = ps.STLSQ(threshold=0.01, alpha=1.0, normalize_columns=True)
#    optimizer = ps.FROLS(alpha=.1)
    optimizer = ps.SR3(threshold=0.01,thresholder="l2", normalize_columns=True,tol=1e-9)
#    optimizer = ps.SSR(alpha=.1)

# Modeling:
    model = ps.SINDy(feature_names=train_df[['x','y','z','t']].columns,\
                     feature_library=my_library,\
                     optimizer=optimizer, t_default=300
                     )
    model.fit(train_df[['x','y','z','t']].values, t=t_train, x_dot=vel_train.values)
#    model.fit(trainer.values, t=t_train)

# Scoring:
    train_days = round(len(train_df)/288,2)
    test_days = round(len(test_df)/288,2)
    print('\n')
    print('Space Vehicle:  PRN ',args.prn)
    print(f'Size of training set: {train_days} days.')
    print(f'Size of test set: {test_days} days.\n')
    print('\nModel (derived using pysindy):')
    model.print()
    score = model.score(test_df[['x','y','z','t']].values, t=t_test,\
                        x_dot=vel_test.values,\
                            metric=sklearn.metrics.mean_squared_error)
    score_km = round(score,4)
    score_m = round(score*1000,4)
    score_cm = round(score*km_to_cm,4)
    print(f'\nModel Score (MSE): {score_km} km')
    print(f'\nModel Score (MSE): {score_m} m')
    print(f'\nModel Score (MSE): {score_cm} cm')
    # x0_test = np.asarray(test_df.iloc[0][['x','y','z','t']])
    # test_sim = model.simulate(x0_test, t_test)  # Takes forever then craps out due to NAN's or something
    
 
#df_train = 