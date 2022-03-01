#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:10:14 2022
  sp3_utils.py:  parsing NGA SP3 files
@author: mcarroll
"""

import re
import pandas as pd
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
        