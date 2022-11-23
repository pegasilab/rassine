# Quickstart

This guide will take you quickly to your first processing. We assume all the commands below are run
in a new empty folder.

These instructions will install the latest RASSINE release. To work with the latest development
version, follow the instructions [here](dev/quickstart.md).

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
source .venv/bin/activate # activate the virtual environment, run it as needed
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
tools enumerated [here](cli).

In addition, you will need to download a few scripts:

- the [run_rassine.sh](https://github.com/pegasilab/rassine/blob/master/run_rassine.sh) script
  that links the different steps of the pipeline together,
- the [GNU Parallel] Perl script if GNU Parallel is not installed on your computer.

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

```bash
curl -L -O https://github.com/pegasilab/HD110315/archive/refs/heads/master.zip
unzip master.zip
```



Run the pipeline. You can look at the `HD110315-master/data/s1d/HARPN/STACKED/` directory to
inspect progress, especially during the "rassine" normalization stage

```bash
./run_rassine.sh -c harpn.ini HD110315-master/data/s1d/HARPN
```

## How to customize the above for a different star

Doc sur ce qu'il va chercher
```{command-output} ../../run_rassine.sh
```


