#!/bin/bash

if [ ! -d spectra_library ]; then
    mkdir spectra_library
fi

if [ ! -f spectra_library/download_successful ]; then
    echo Downloading sample spectra
    curl -L -o spectra_library/HD23249.tar.gz https://www.dropbox.com/s/c5e69deuia2wc5b/HD23249.tar.gz?dl=0
    curl -L -o spectra_library/HD110315.tar.gz https://www.dropbox.com/s/g8f8q40a66vqbw6/HD110315.tar.gz?dl=0
    touch spectra_library/download_successful
else
    echo Reusing already downloaded sample data
fi

if [ -d spectra_library/HD110315 ]; then
    echo Removing HD110315 data
    rm -rf spectra_library/HD110315
fi
if [ -d spectra_library/HD23249 ]; then
    echo Removing HD23249 data
    rm -rf spectra_library/HD23249
fi
echo Decompressing HD110315
tar xfz spectra_library/HD110315.tar.gz -C spectra_library
echo Decompressing HD23249
tar xfz spectra_library/HD23249.tar.gz -C spectra_library
