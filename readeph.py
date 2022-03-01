#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
readeph.py
A script to read and parse NGA precise ephemeris files, i.e., SP3 files:
    e.g., nga21906.eph
and produce csv files in which the position and velocity records are combined into 
single timestamped records.
Created on Tue Mar  1 09:53:11 2022

@author: mcarroll
"""

#import re
import sp3_utils as su
import argparse as ap
import glob
import sys

parser = ap.ArgumentParser()
parser.add_argument('infile', help='Input file:  Required!!!')
parser.add_argument('--path', help='Path to folder with EPH files.  Optional')
parser.add_argument('--outfile','-of', help='Output file:  Optional.  Default is stdout.')
args = parser.parse_args()

#ephfile = 'NGA/nga21906.eph'
if args.infile=='all' or args.infile=='*':  # getting all .eph files in path
    if args.path is not None:
        filespec = f'{args.path}/nga*.eph'
        filelist = glob.glob(filespec)
    else:
        filelist = glob.glob('nga*.eph')  # assumes CWD
else:
    filelist = args.infile.split(',')  

if filelist==[]:
    print('Empty file list!!! Consider using --path.')
    sys.exit()
else:
    filelist.sort()

if args.outfile is not None:
    out = open(args.outfile,'wt')
    out.write('recnum,timestamp,PRN,x,y,z,t,v_x,v_y,v_z,v_t\n')

mismatch_count = 0
rec_count = 0
file_count = 0

for file in filelist:
    fp = open(file,'rt')
    file_count += 1
    line = fp.readline()
    while line:
    # parse timestamp lines:
        if line.find('*')==0:  # Means this is an epoch date and time
            timestamp_str,_ = su.parse_timestamp(line)
        elif line.find('P ')==0:
            p_rec = line
        elif line.find('V ')==0:
            v_rec = line
            result, csvrec = su.gen_csvrec(timestamp_str,p_rec,v_rec)
            if result:
                if args.outfile is not None:
                    out.write(str(rec_count) +',')
                    out.write(csvrec+'\n') # include newline!!!
                else:
                    print(csvrec)
                rec_count += 1
            else:
                mismatch_count +=1
        line = fp.readline()
    fp.close()
out.close()
if mismatch_count:
    print('Mismatch Count:',mismatch_count)

    