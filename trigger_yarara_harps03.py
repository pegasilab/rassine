#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:49:52 2020

@author: Cretignier Michael 
@university University of Geneva
"""

import sys
import os 
import matplotlib.pylab as plt
import numpy as np
# import my_rassine_tools as myr
import getopt
import glob as glob
import time
import pandas as pd 
import astropy.coordinates as astrocoord
import pickle
from astropy import units as u

# =============================================================================
# TABLE
# =============================================================================
#
# stage = -2 (preprocessing importation from dace)
# stage = -1 (launch trigger RASSINE)
#
#
# =============================================================================


# =============================================================================
# PARAMETERS
# =============================================================================

star = 'HD20794'
ins = 'HARPS03'
input_product = 's1d'

stage = 1000          #begin at 1 when data are already processed 
stage_break = 28    #break included
cascade = True
close_figure = True
planet_activated = False
rassine_full_auto = 0
bin_length = 1
fast = False
reference = None
verbose=2
prefit_planet = False
drs_version = 'old'
sub_dico_to_analyse = 'matching_diff'
m_clipping = 3


if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:i:b:e:c:p:a:l:r:f:d:v:k:D:S:m:')
    for j in optlist:
        if j[0] == '-s':
            star = j[1]
        elif j[0] == '-i':
            ins = j[1]
        elif j[0] == '-b':
            stage = int(j[1])
        elif j[0] == '-e':
            stage_break = int(j[1])
        elif j[0] == '-c':
            cascade = int(j[1])      
        elif j[0] == '-p':
            planet_activated = int(j[1])  
        elif j[0] == '-a':
            rassine_full_auto = int(j[1]) 
        elif j[0] == '-l':
            bin_length = int(j[1])
        elif j[0] == '-r':
            reference = j[1]        
        elif j[0] == '-f':
            fast = int(j[1]) 
        elif j[0] == '-d':
            close_figure = int(j[1]) 
        elif j[0] == '-v':
            verbose = int(j[1])         
        elif j[0] == '-k':
            prefit_planet = int(j[1]) 
        elif j[0] == '-D':
            drs_version = j[1] 
        elif j[0] == '-S':
            sub_dico_to_analyse = j[1]         
        elif j[0] == '-m':
            m_clipping = int(j[1])
            
cwd = os.getcwd()
root = cwd
directory_yarara = root+'/spectra_library/'
directory_to_dace = directory_yarara + star + '/data/'+input_product+'/spectroDownload'
directory_rassine = '/'.join(directory_to_dace.split('/')[0:-1])+'/'+ins
directory_reduced_rassine = directory_rassine + '/STACKED/'
directory_workspace = directory_rassine + '/WORKSPACE/'

# =============================================================================
# BEGINNING OF THE TRIGGER
# =============================================================================

def print_iter(verbose):
    from colorama import Fore
    if verbose==-1:
        print(Fore.BLUE + " ==============================================================================\n [INFO] Extracting data with RASSINE...\n ==============================================================================\n"+ Fore.RESET)
    elif verbose==0:
        print(Fore.BLUE + " ==============================================================================\n [INFO] Preprocessing data with RASSINE...\n ==============================================================================\n"+ Fore.RESET)
    elif verbose==1:
        print(Fore.GREEN + " ==============================================================================\n [INFO] First iteration is beginning...\n ==============================================================================\n"+ Fore.RESET)
    elif verbose==2:
        print(Fore.YELLOW + " ==============================================================================\n [INFO] Second iteration is beginning...\n ==============================================================================\n"+ Fore.RESET)
    elif verbose==42:
        print(Fore.YELLOW + " ==============================================================================\n [INFO] Merging is beginning...\n ==============================================================================\n"+ Fore.RESET)
    else:
        hours = verbose // 3600 % 24
        minutes = verbose // 60 % 60
        seconds = verbose % 60
        print(Fore.RED + " ==============================================================================\n [INFO] Intermediate time : %.0fh %.0fm %.0fs \n ==============================================================================\n"%(hours,minutes,seconds)+ Fore.RESET)

print_iter(verbose)

obs_loc = astrocoord.EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) #HARPN

begin = time.time()
all_time= [begin]

time_step = {'begin':0}
button=0
instrument = ins#[0:5]

def get_time_step(step):
    now = time.time()
    time_step[step] = now - all_time[-1]
    all_time.append(now)

def break_func(stage):
    button=1
    if stage>=stage_break:
        return 99
    else:
        if cascade:
            return stage + 1
        else:
            return stage
        
if stage==-1:
    
    # =============================================================================
    # RUN RASSINE TRIGGER
    # =============================================================================
    
    #in topython terminal, change the Trigger file 
    print(' python Rassine_trigger.py -s %s -i %s -a %s -b %s -d %s -l 0.01 -o %s'%(star, [instrument[0:5],'ESPRESSO'][drs_version!='old'], str(rassine_full_auto), str(bin_length), ins, directory_rassine+'/'))
    os.system('python Rassine_trigger.py -s %s -i %s -a %s -b %s -d %s -l 0.01 -o %s'%(star, [instrument[0:5],'ESPRESSO'][drs_version!='old'], str(rassine_full_auto), str(bin_length), ins, directory_rassine+'/'))
    get_time_step('rassine')
    
    reduced_files = glob.glob(directory_reduced_rassine+'RASSINE*.p')

    if len(reduced_files):   
        if not os.path.exists(directory_workspace):
            os.system('mkdir '+directory_workspace)
            
        for k in reduced_files:
            os.system('cp '+k+' '+directory_workspace)              

    stage = break_func(stage)
