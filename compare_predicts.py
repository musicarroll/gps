#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:38:16 2022
    compare_predicts.py
    Compares NGA 9-day predicts against NGA public truth (Precise Ephemeris=PE)
    after the fact.
    Data derived from NGA Geomatics downloads from
    https://earth-info.nga.mil/index.php?dir=gnss&action=gnss
    Github site of pre-processed data: https://github.com/musicarroll/gps
@author: mcarroll
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sp3_utils as su
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('infile', help='Input file:  Required!!!')
parser.add_argument('predicts', help='Predicts input file.')
parser.add_argument('--prn', help='Specific PRN to look at.')
args = parser.parse_args()

km_to_cm = 100*1000

if args.infile is not None:
    df = su.get_df(args.infile)    
if args.predicts is not None:
    predicts = su.get_df(args.predicts)
 
pts = np.array(predicts['timestamp'])
filt = df['timestamp'].isin(pts)
df_truth = df[filt]
if args.prn is not None:
    prnfilt = df_truth['PRN']==int(args.prn)
    prn_df_truth = df_truth[prnfilt]
    predict_prn_filter = predicts['PRN']==int(args.prn)
    predict_prn_df = predicts[predict_prn_filter]  # assert lengths of these are equal
    rtrue = np.array(prn_df_truth['r'])
    rpred = np.array(predict_prn_df['r'])
    diff= su.get_vec_diff(predict_prn_df, prn_df_truth)
    delta = np.linalg.norm(diff,axis=1)
    epochs = np.arange(0,len(prn_df_truth))
    fig,axs = plt.subplots(2,1)
    fig.suptitle('Comparison of NGA Truth and NGA Predicts \nPosition Vectors over 9 Days')
    axs[0].plot(epochs,rtrue,label='true r')
    axs[0].plot(epochs,rpred, label='predicted r')
    axs[0].set_ylabel('km')
    axs[0].legend()
    axs[1].plot(epochs,delta*km_to_cm,label='delta r = norm(predicted r - true r)')
    axs[1].set_ylabel('cm')
    axs[1].legend()
    axs[1].set_xlabel('15m epochs')
    plt.show()
    
    
    
    
    