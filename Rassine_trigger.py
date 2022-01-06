#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:50:04 2019

@author: cretignier
"""

from __future__ import print_function
import matplotlib
import platform 

if platform.system()=='Linux':
    matplotlib.use('Agg',force=True)
else:
    matplotlib.use('Qt5Agg',force=True)

import glob as glob 
import Rassine_functions as ras      
import numpy as np 
import matplotlib.pylab as plt
import os
import sys
import pandas as pd
import getopt

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

# =============================================================================
# parameters
# =============================================================================

instrument = 'HARPN'                                                     # instrument (either HARPS, HARPN, CORALIE or ESPRESSO for the moment)
dir_spec_timeseries = root+'/Yarara/HD4628/data/s1d/HARPN/'   # directory containing the s1d spectra timeseries

nthreads_preprocess = 6               # number of threads in parallel for the preprocessing
nthreads_matching = 2                 # number of threads in parallel for the matching (more than 2 is not efficient for some reasons...) 
nthreads_rassine = 4                  # number of threads in parallel for the normalisation (BE CAREFUL RASSINE NEED A LOT OF RAM DEPENDING ON SPECTRUM LENGTH)
nthreads_intersect = 6                # number of threads in parallel for the post-continuum fit 

rv_timeseries = root+'/Yarara/HD4628/data/s1d/HARPN/DACE_TABLE/Dace_extracted_table.csv'      # RV time-series to remove in kms, (only if binaries with ~kms RV amplitudes) stored in a pickle dictionnary inside 'model' keyword 

dlambda = None                     # wavelength grid step of the equidistant grid, only if unevenly wavelength grid on lack of homogeneity in spectra time-series
bin_length_stack = 1               # length of the binning for the stacking in days
dbin = 0                           # dbin to shift the binning (0.5 for solar data)
use_master_as_reference = False    # use master spectrum as reference for the clustering algorithm (case of low SNR spectra)
full_auto = False  

# =============================================================================
# buttons
# =============================================================================

preprocessed = 1
match_frame = 1
stacking = 1
rassine_normalisation_master = 1
rassine_normalisation = 1
rassine_intersect_continuum = 1
rassine_diff_continuum = 1

# =============================================================================
# trigger
# =============================================================================

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:i:a:b:d:l:o:')
    for j in optlist:
        if j[0] == '-s': 
            star = j[1]
        if j[0] == '-i': 
            instrument = j[1]
        if j[0] == '-a': 
            full_auto = bool(int(j[1]))
        if j[0] == '-b': 
            bin_length_stack = int(j[1])
        if j[0] == '-l': 
            dlambda = float(j[1])
        if j[0] == '-d': 
            directory = j[1] 
        if j[0] == '-o': 
            output_dir = j[1]

    dir_spec_timeseries = root+'/Yarara/'+star+'/data/s1d/'+directory+'/'   
    rv_timeseries = root+'/Yarara/'+star+'/data/s1d/'+directory+'/DACE_TABLE/Dace_extracted_table.csv'

try:
    print(' Output directory is :', output_dir)
except:
    output_dir =  dir_spec_timeseries

file_ver = ''
if os.getcwd()=='/Users/cretignier/Documents/Python':
    file_ver = '3.8'
    print('Switch for the version multiprocessing 3.8')

if preprocessed:
    print(' [STEP INFO] Preprocessing...')
    if os.path.exists(rv_timeseries):
        print(' [INFO] Table correctly specified')
        os.system('python Rassine'+file_ver+'_multiprocessed.py -v PREPROCESS -s '+str(rv_timeseries)+' -n '+str(nthreads_preprocess)+' -i '+instrument+' -o '+output_dir) #reduce files in the fileroot column on the dace table 
    else:
        print(' [INFO] Table uncorrectly specified : %s'%(rv_timeseries))
        os.system('python Rassine'+file_ver+'_multiprocessed.py -v PREPROCESS -s '+dir_spec_timeseries+' -n '+str(nthreads_preprocess)+' -i '+instrument+' -o '+output_dir)

    if not len(glob.glob(dir_spec_timeseries+'PREPROCESSED/*.p')):
        error = sys.exit(' [ERROR] No file found in the preprocessing directory : %s'%(dir_spec_timeseries+'PREPROCESSED/'))

if match_frame:
    print(' [STEP INFO] Matching frame...')
    os.system('python Rassine'+file_ver+'_multiprocessed.py -v MATCHING -s '+dir_spec_timeseries+'PREPROCESSED/'+' -n '+str(nthreads_matching)+' -d '+str(dlambda)+' -k '+str(rv_timeseries))

if not os.path.exists(dir_spec_timeseries+'MASTER/'):
    os.system('mkdir '+dir_spec_timeseries+'MASTER/')

if stacking:
    print(' [STEP INFO] Stacking frame...')
    master_name = ras.preprocess_stack(glob.glob(dir_spec_timeseries+'PREPROCESSED/*.p'), 
                         bin_length = bin_length_stack, 
                         dbin = dbin,
                         make_master=True)

    previous_files = glob.glob(dir_spec_timeseries+'MASTER/*Master_spectrum*.p')
    for file_to_delete in previous_files:
        os.system('rm '+file_to_delete)
    
    os.system('mv '+dir_spec_timeseries+'STACKED/'+master_name+' '+dir_spec_timeseries+'MASTER/')
    print(' Master file %s displaced to %s'%(master_name, dir_spec_timeseries+'MASTER/'))

if os.path.exists(dir_spec_timeseries+'PREPROCESSED/'):
    os.system('rm -rf '+dir_spec_timeseries+'PREPROCESSED/')

master_name = glob.glob(dir_spec_timeseries+'MASTER/Master_spectrum*.p')[0]
if rassine_normalisation_master:
    print(' [STEP INFO] Normalisation master spectrum')
    os.system('python Rassine.py -s '+master_name+' -a '+str(bool(1-full_auto))+' -S True -e '+str(bool(1-full_auto)))


master_name = glob.glob(dir_spec_timeseries+'MASTER/RASSINE_Master_spectrum*.p')[0]
if rassine_normalisation:
    print(' [STEP INFO] Normalisation frame...')
    os.system('python Rassine'+file_ver+'_multiprocessed.py -v RASSINE -s '+dir_spec_timeseries+'STACKED/Stacked -n '+str(nthreads_rassine)+' -l '+master_name+' -P '+str(True)+' -e '+str(False))


master_spectrum_name = None
if use_master_as_reference:
    master_spectrum_name = master_name
    
if rassine_intersect_continuum:
    print(' [STEP INFO] Normalisation clustering algorithm')
    ras.intersect_all_continuum_sphinx(glob.glob(dir_spec_timeseries+'STACKED/RASSINE*.p'), feedback = bool(1-full_auto),  master_spectrum=master_spectrum_name, copies_master=0, kind='anchor_index', nthreads=6, fraction=0.2, threshold = 0.66, tolerance=0.5, add_new = True)

    os.system('python Rassine'+file_ver+'_multiprocessed.py -v INTERSECT -s '+dir_spec_timeseries+'STACKED/RASSINE -n '+str(nthreads_intersect))


if rassine_diff_continuum:
    print(' [STEP INFO] Normalisation diff continuum')
    if bool(full_auto):
        savgol_window = 200
        master_file = master_name
    else:
        master_file, savgol_window = ras.matching_diff_continuum_sphinx(glob.glob(dir_spec_timeseries+'STACKED/RASSINE*.p'), master=master_name, sub_dico = 'matching_anchors', savgol_window = 200, zero_point=False)

    error = os.system('python Rassine'+file_ver+'_multiprocessed.py -v SAVGOL -s '+dir_spec_timeseries+'STACKED/RASSINE -n '+str(nthreads_intersect)+' -l '+master_name+' -P '+str(True)+' -e '+str(False)+' -w '+str(savgol_window))

    if not error:
        os.system('touch '+dir_spec_timeseries+'rassine_finished.txt')
    
    #ras.matching_diff_continuum(glob.glob(dir_spec_timeseries+'STACKED/RASSINE*.p'), sub_dico = 'matching_anchors', savgol_window = savgol_window, zero_point=False)

plt.close('all')

ras.make_sound('Thank you and see you soon')