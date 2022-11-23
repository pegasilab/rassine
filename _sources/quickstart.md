# Quickstart

This guide will take you quickly to your first processing. We assume all the commands below are run
in a new empty folder.

These instructions will install the latest RASSINE release. To work with the latest development
version, follow the instructions [here](dev/quickstart).

## Python version and fresh virtual environment

Before starting, verify your Python version is at least 3.8. If not, you may want to install a
different Python interpreter using [Conda](https://docs.conda.io/en/latest/) or
[pyenv](https://github.com/pyenv/pyenv).

```bash
python --version
```

We now create a fresh virtual environment. This avoids the RASSINE installation changing packages
on your main Python installation when it brings its required dependencies.

If you are using vanilla Python, type the following commands:

```bash
python -m venv .venv # create a fresh virtual environment, run it only once
source .venv/bin/activate # activate the virtual environment, run it every time you restart your shell/computer
```

If you are using Conda, use something like:

```bash
conda create --name rassinetest python=3.8
conda activate rassinetest
```

## Install RASSINE, helper scripts and configuration files

Simply use `pip`.

```bash
pip install rassine
```

`pip` installs the packages needed by RASSINE, the `rassine` Python package, and the command-line
tools enumerated [here](cli/index.rst).

In addition, you will need to download a few scripts:

- the [run_rassine.sh](https://github.com/pegasilab/rassine/blob/master/run_rassine.sh) script
  that links the different steps of the pipeline together,
- the [GNU Parallel](https://www.gnu.org/software/parallel/) Perl script if GNU Parallel is not installed on your computer.

We will download those files in our folder, and set their executable flag.
As an alternative, you could put those two scripts somewhere in your path.

You can skip the first two lines below if GNU Parallel is already installed on your computer.

```bash
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/parallel
chmod +x parallel
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/run_rassine.sh
chmod +x run_rassine.sh
```

We will also download the configuration files for the HARPN and HARPS03 instruments.

```bash
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/harpn.ini
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/harps03.ini
```

## Download and unzip HD110315 observations

This quickstart guide will work with the HD110315 data.

```bash
curl -L -o HD110315-master.zip https://github.com/pegasilab/HD110315/archive/refs/heads/master.zip
unzip HD110315-master.zip
```

We now run the pipeline, which takes a few minutes.

```bash
./run_rassine.sh -c harpn.ini HD110315-master/data/s1d/HARPN
```

Check the results in the `HD110315-master/data/s1d/HARPN/STACKED` directory.

## How to customize the above for a different star

To run the pipeline for a different star, you have to:

1. Provide the relevant raw data. In the path above, the `STARNAME/data/s1d/INSTRUMENTNAME` prefix
   is a convention. RASSINE does not require you to use that convention. However, in the folder, you
   have to provide two sets of data. First, scalar information about the observation in
  `DACE_TABLE/Dace_extracted_table.csv`. The required columns are explained [here](_autosummary/rassine.imports.data.DACE).
  In the `RAW/` subdirectory, you have to provide the `fits` files. Follow the
  `INSTRUMENT.DATE_s1d_A.fits` convention as in the examples we provide.

2. Provide the correct configuration file. In the current release, we provide the 
   [HARPN](https://github.com/pegasilab/rassine/blob/master/harpn.ini) and 
   [HARPS03](https://github.com/pegasilab/rassine/blob/master/harps03.ini) configuration files.

For the HD23249 star, the steps look like:

```bash
curl -L -o HD23249-master.zip https://github.com/pegasilab/HD23249/archive/refs/heads/master.zip
unzip HD23249-master.zip
./run_rassine.sh -c harps03.ini HD23249-master/data/s1d/HARPS03
```

## The `run_rassine.sh` parameters

```{command-output} ../../run_rassine.sh -h
```