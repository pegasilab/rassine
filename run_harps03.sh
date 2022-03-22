#!/bin/bash
set -u

nthreads=4 # number of concurrent jobs
nchunks=40 # number of spectra per Python script invocation

dlambda=0.01

## Preprocess
# Step 1: we read the FITS files in the format of a particular instrument and output files in the format
# expected by RASSINE. The data is reformatted, and a few parameters are extracted
#
# This step did correspond to
# python Rassine_multiprocessed.py -v PREPROCESS -s "$dace_table" -n $nthreads_preprocess -i HARPS -o "$output_dir"
export RASSINE_ROOT=/home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03
export RASSINE_CONFIG=harps03.ini
mkdir -p $RASSINE_ROOT/PREPROCESSED
#check_and_enumerate -i raw/ | parallel --will-cite --verbose -N$nchunks --jobs $nthreads --keep-order \
#  rassine_preprocess --input-folder raw/ --output-folder PREPROCESSED/
borders_scan -i PREPROCESSED/ -o matching.h5 --dlambda 0.01 --apply-rv-shift true
exit
## Match frame (Borders)
# This was python Rassine_multiprocessed.py -v MATCHING -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED/ -n $nthreads_matching -d $dlambda -k /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/DACE_TABLE/Dace_extracted_table.csv
# Steo 2A we extract the frame parameters and write them to a file
matching_parameters=/home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/matching.h5
preprocessed_dir=/home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED

# TODO: -> rassine_borders_scan.py
# TODO: -> rassine_borders_reinterpolate.py

python rassine_borders_scan.py --input-dir "$preprocessed_dir" --dlambda 0.01 -k "$dace_table" --output-file "$matching_parameters"
exit
# Step 2B we process each spectrum in parallel
# files are overwritten

nchunks_matching=10
nthreads_matching=4

nspectra=$(ls "$preprocessed_dir"/*.p | wc -l)
seq 0 $(($nspectra - 1)) | parallel --will-cite -N$nchunks_matching --jobs $nthreads_matching --keep-order python rassine_borders_reinterpolate.py --parameter-file "$matching_parameters" --input-dir "$preprocessed_dir"

## Stacking
# Step 3
# this is a temporal summation to reduce the noise; sum instead of average as to keep the error estimation
# this creates the MASTER file which sums all the spectra
# The output files are written in /STACKED
mkdir -p /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER
python rassine_stacking.py --input_directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED

# get the latest master file
master=$(ls -t /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Master*.p | head -n 1)
# TODO: safe Bash practices
rm /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER/*Master_spectrum*.p
mv $master /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER/
rm /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED/*

## RASSINE normalization
# This was python Rassine_multiprocessed.py -v RASSINE -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Stacked -n $nthreads_rassine -l "$rassine_master" -P True -e False
# First we process the Master file to obtain the RASSINE_Master* file
# This computes the parameters of the model

# TODO: -> rassine.py

not_full_auto=1 # we get different results by using the GUI and just pressing ENTER vs. full auto mode, is that bad?
master=$(ls -t /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER/Master*.p | head -n 1)
python rassine_main.py -s $master -a $not_full_auto -S True -e $not_full_auto
rassine_master=$(ls -t /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER/RASSINE_Master_spectrum*.p | head -n 1)

nthreads_rassine=4
# Step 4B Normalisation frame, done in parallel
output_dir=/home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED
ls /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Stacked*.p | parallel --will-cite --jobs $nthreads_rassine --keep-order python rassine_main.py -o $output_dir -a False -P True -e False -l $rassine_master -s


## Intersect continuum
# This was python Rassine_multiprocessed.py -v INTERSECT -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect

# Step 5A: computation of the parameters
# See Fig D7
nthreads_intersect1=4
python rassine_anchor_scan.py --input-directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED --feedback $not_full_auto --no-master-spectrum --copies-master 0 --kind anchor_index --nthreads $nthreads_intersect1 --fraction 0.2 --threshold 0.66 --tolerance 0.5 --add-new True

# Step 5B: application in parallel
# Done in place
nthreads_intersect2=4
nchunks_intersect2=10

ls /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE* | parallel --will-cite  -N$nchunks_intersect2 --jobs $nthreads_intersect2 --keep-order python rassine_anchor_filter.py
# note: removed the master spectrum option

## matching_diff_continuum_sphinx

# Step 6A
# See Fig D8 of the RASSINE paper
output_savgol_window=/home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/savgol_window.txt
python rassine_savgol_gui.py --input_directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED --output-file $output_savgol_window --no-master
savgol_window=$(< $output_savgol_window)

# Step 6B done in parallel
# "matching_diff"
nthreads_savgol=4
nchunks_savgol=10
ls /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE* | parallel --will-cite  -N$nchunks_savgol --jobs $nthreads_savgol --keep-order python rassine_savgol_filter.py --anchor-file $rassine_master --savgol-window $savgol_window
#python Rassine_multiprocessed.py -v SAVGOL -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect -l $rassine_master -P True -e False -w $savgol_window
touch /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/rassine_finished.txt


# TODO:
# 1) Split the instrument preprocessing in different scripts
# 2) Make all paths visible in the options
# 3) Introduce a configuration file format
# 4) Introduce a class describing the RASSINE input, or find a way to describe the input data
# 5) Introduce a class describing the RASSINE output, or find a way to describe the output data
# 6) Introduce functional tests for each step, or bunch of steps
# 7) See if rassine_intersect1.py would be faster using HDF5 vs pickle
# 8) Time all the steps
# 9) Code coverage, plus modularization of functions in the Python package
