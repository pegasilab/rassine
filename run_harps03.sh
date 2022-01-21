#!/bin/bash
set -u
# number of threads in parallel for the preprocessing
nthreads_preprocess=6
nchunks_preprocess=40
# number of threads in parallel for the matching (more than 2 is not efficient for some reasons...)
nthreads_matching=6
nchunks_matching=40
# number of threads in parallel for the normalisation (BE CAREFUL RASSINE NEED A LOT OF RAM DEPENDING ON SPECTRUM LENGTH)
nthreads_rassine=4
# number of threads in parallel for the post-continuum fit
nthreads_intersect=6
dlambda=0.01
## Preprocess
dace_table=/home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/DACE_TABLE/Dace_extracted_table.csv

output_dir=/home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/
python extract_fileroot_from_dace_table.py "$dace_table" | parallel --eta -N$nchunks_preprocess --jobs $nthreads_preprocess --keep-order python rassine_preprocess.py -i HARPS -o "$output_dir"

#for inputfile in $(python extract_fileroot_from_dace_table.py "$dace_table")
#do
#  echo $inputfile
#  python rassine_preprocess.py $inputfile -i HARPS -o /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/
#done

#python Rassine_multiprocessed.py -v PREPROCESS -s "$dace_table" -n $nthreads_preprocess -i HARPS -o "$output_dir"

## Match frame
matching_parameters=/home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/matching.h5
preprocessed_dir=/home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED
python rassine_preprocess_match_stellar_frame.py --input-dir "$preprocessed_dir" --dlambda 0.01 -k "$dace_table" --output-file "$matching_parameters"
nspectra=$(ls "$preprocessed_dir"/*.p | wc -l)
seq 0 $(($nspectra - 1)) | parallel --eta -N$nchunks_matching --jobs $nthreads_matching --keep-order python rassine_matching.py --parameter-file "$matching_parameters" --input-dir "$preprocessed_dir"
# python Rassine_multiprocessed.py -v MATCHING -s /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED/ -n $nthreads_matching -d $dlambda -k /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/DACE_TABLE/Dace_extracted_table.csv
## Stacking

mkdir -p /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/MASTER
python rassine_stacking.py --input_directory /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED
# get the latest master file
master=$(ls -t /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Master*.p | head -n 1)
rm /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/MASTER/*Master_spectrum*.p
mv $master /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/MASTER/
rm /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/PREPROCESSED/*
exit
not_full_auto=0
master=$(ls -t /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/MASTER/Master*.p | head -n 1)
python Rassine.py -s $master -a $not_full_auto -S True -e $not_full_auto
rassine_master=$(ls -t /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/MASTER/RASSINE_Master_spectrum*.p | head -n 1)

# Normalisation frame
python Rassine_multiprocessed.py -v RASSINE -s /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED/Stacked -n $nthreads_rassine -l "$rassine_master" -P True -e False
# Intersect continuum
# note: removed the master spectrum option
python rassine_intersect.py --input_directory /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED --feedback $not_full_auto --no-master_spectrum --copies_master 0 --kind anchor_index --nthreads $nthreads_intersect --fraction 0.2 --threshold 0.66 --tolerance 0.5 --add_new True
python Rassine_multiprocessed.py -v INTERSECT -s /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect
# matching_diff_continuum_sphinx
#savgol_window=$(python rassine_matching_diff_continuum_sphinx.py --input_directory /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED --no-master | tail -n 1)
#echo $savgol_window
savgol_window=200
python Rassine_multiprocessed.py -v SAVGOL -s /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/STACKED/RASSINE -n $nthreads_intersect -l $rassine_master -P True -e False -w $savgol_window
touch /home/denis/w/rassine/spectra_library/HD23249/data/s1d/HARPS03/rassine_finished.txt
