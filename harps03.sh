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

if [ ! -f $RASSINE_ROOT/tag ]
then
# create a new tag
date -u +%Y-%m-%dT%H:%M:%S > $RASSINE_ROOT/tag
fi

tag=$(<$RASSINE_ROOT/tag)
master_table="Master_spectrum_$tag.p"

if [ $# -eq 0 ] || [ $1 == "import" ]
then

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
fi


if [ $# -eq 0 ] || [ $1 == "reinterpolate" ]
then
# if the summary table exists, remove it
individual_reinterpolated=$RASSINE_ROOT/individual_reinterpolated.csv
[ -f $individual_reinterpolated ] && rm $individual_reinterpolated 
enumerate_table_rows individual_imported.csv | parallel $PARALLEL_OPTIONS \
  reinterpolate -i PREPROCESSED -o PREPROCESSED -I individual_imported.csv -O individual_reinterpolated.csv
reorder_csv --column name --reference individual_imported.csv individual_reinterpolated.csv 
fi


if [ $# -eq 0 ] || [ $1 == "stacking" ]
then

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

sort_csv --column group stacked_basic.csv
#mkdir -p /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/MASTER
#python rassine_stacking.py --input_directory /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED

stacking_master_spectrum -I stacked_basic.csv -O master_spectrum.csv -i STACKED -o MASTER/$master_table
fi

if [ $# -eq 0 ] || [ $1 == "rassine" ]
then

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
  rassine --input-table stacked_basic.csv --input-folder STACKED --output-folder STACKED --config $RASSINE_ROOT/anchor_Master_spectrum_$tag.ini --output-folder STACKED --output-plot-folder STACKED

fi


if [ $# -eq 0 ] || [ $1 == "matching_anchors" ]
then
## Intersect continuum
# This was python Rassine_multiprocessed.py -v INTERSECT -s /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect

# Step 5A: computation of the parameters
# See Fig D7

# rm /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/STACKED
matching_anchors_scan --input-table stacked_basic.csv --input-pattern STACKED/RASSINE_{name}.p --output-file MASTER/Master_tool_$tag.p --no-master-spectrum --copies-master 0  --fraction 0.2 --threshold 0.66 --tolerance 0.5 

# Step 5B: application in parallel
# Done in place

anchors_table=$RASSINE_ROOT/matching_anchors.csv
[ -f $anchors_table ] && rm $anchors_table 

enumerate_table_rows stacked_basic.csv | parallel $PARALLEL_OPTIONS \
  matching_anchors_filter --master-tool MASTER/Master_tool_$tag.p --process-table stacked_basic.csv --process-pattern STACKED/RASSINE_{name}.p --output-table matching_anchors.csv 
reorder_csv --column name --reference stacked_basic.csv matching_anchors.csv 

# process master last
matching_anchors_filter --master-tool MASTER/Master_tool_$tag.p --process-master MASTER/RASSINE_$master_table --output-table matching_anchors.csv 

fi


if [ $# -eq 0 ] || [ $1 == "matching_diff" ]
then

# Step 6B done in parallel
# "matching_diff"
enumerate_table_rows stacked_basic.csv | parallel $PARALLEL_OPTIONS \
  matching_diff --anchor-file MASTER/RASSINE_$master_table --savgol-window 200 --process-table stacked_basic.csv --process-pattern STACKED/RASSINE_{name}.p
touch /home/denis/w/rassine1/spectra_library/HD23249/data/s1d/HARPS03/rassine_finished.txt

fi

rm $RASSINE_ROOT/*.lock

# TODO:
# 1) Split the instrument preprocessing in different scripts
# 2) Make all paths visible in the options
# 3) Introduce a configuration file format
# 7) See if rassine_intersect1.py would be faster using HDF5 vs pickle
# 8) Time all the steps
