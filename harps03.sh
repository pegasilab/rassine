#!/bin/bash
set -u

script_path=$(realpath "${BASH_SOURCE[-1]}")
script_folder=$(dirname "$script_path")
export RASSINE_ROOT=$script_folder/spectra_library/HD23249/data/s1d/HARPS03
export RASSINE_CONFIG=$script_folder/harps03.ini
export RASSINE_LOGGING_LEVEL=INFO
nthreads=4 # number of concurrent jobs
nchunks=10 # number of items per Python script invocation

PARALLEL_OPTIONS="--will-cite --verbose -N$nchunks --jobs $nthreads --keep-order"

##
## Preprocess
##
# We read the FITS files in the format of a particular instrument and output files in the format
# expected by RASSINE. The data is reformatted, and a few parameters are extracted

preprocess_table -I DACE_TABLE/Dace_extracted_table.csv -i raw -O individual_basic.csv

# delete the produced table if it exists already
individual_imported=$RASSINE_ROOT/individual_imported.csv
[ -f $individual_imported ] && rm $individual_imported 
enumerate_table_rows individual_basic.csv | parallel $PARALLEL_OPTIONS \
  preprocess_import -i raw -o PREPROCESSED -I individual_basic.csv -O individual_imported.csv
reorder_csv --column name --reference individual_basic.csv individual_imported.csv 

# if the summary table exists, remove it
individual_reinterpolated=$RASSINE_ROOT/individual_reinterpolated.csv
[ -f $individual_reinterpolated ] && rm $individual_reinterpolated 
enumerate_table_rows individual_imported.csv | parallel $PARALLEL_OPTIONS \
  reinterpolate -i PREPROCESSED -o PREPROCESSED -I individual_imported.csv -O individual_reinterpolated.csv
reorder_csv --column name --reference individual_imported.csv individual_reinterpolated.csv 


## Stacking
# Step 3
# this is a temporal summation to reduce the noise; sum instead of average as to keep the error estimation
# this creates the MASTER file which sums all the spectra
# The output files are written in /STACKED

stacking_create_groups -I individual_reinterpolated.csv -O individual_group.csv
stacked_basic=$RASSINE_ROOT/stacked_basic.csv
[ -f $stacked_basic ] && rm $stacked_basic 

enumerate_table_column_unique_values -c group individual_group.csv | parallel $PARALLEL_OPTIONS \
  stacking_stack -I individual_reinterpolated.csv -G individual_group.csv -O stacked_basic.csv -i PREPROCESSED -o STACKED
#mkdir -p /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER
#python rassine_stacking.py --input_directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED

tag=$(date +%Y-%m-%dT%H:%M:%S)
master_table="Master_spectrum_$tag.p"
stacking_master_spectrum -I stacked_basic.csv -O master_spectrum.csv -i STACKED -o MASTER/$master_table


## RASSINE normalization
# This was python Rassine_multiprocessed.py -v RASSINE -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Stacked -n $nthreads_rassine -l "$rassine_master" -P True -e False
# First we process the Master file to obtain the RASSINE_Master* file
# This computes the parameters of the model

# TODO: -> rassine.py

##
## RASSINE main processing on master file
##

not_full_auto=1 # we get different results by using the GUI and just pressing ENTER vs. full auto mode, is that bad?
rassine --input-spectrum $master_table --input-folder MASTER --output-folder MASTER --output-plot-folder MASTER --output-anchor-ini anchor_Master_spectrum_$tag.ini
# RASSINE_Master has additional stuff
##
## 
##
# Step 4B Normalisation frame, done in parallel
enumerate_table_rows stacked_basic.csv | parallel $PARALLEL_OPTIONS \
  rassine --logging-level WARNING --input-table stacked_basic.csv --input-folder STACKED --output-folder STACKED --config $RASSINE_ROOT/anchor_Master_spectrum_$tag.ini --output-folder STACKED --output-plot-folder STACKED

## Intersect continuum
# This was python Rassine_multiprocessed.py -v INTERSECT -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect

# Step 5A: computation of the parameters
# See Fig D7
nthreads_intersect1=4
python rassine_anchor_scan.py --input-directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED --feedback $not_full_auto --no-master-spectrum --copies-master 0 --kind anchor_index --nthreads $nthreads_intersect1 --fraction 0.2 --threshold 0.66 --tolerance 0.5 --add-new True

exit
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
