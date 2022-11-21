# RASSINE

RASSINE is a tool to ..


Install Poetry https://python-poetry.org/

```
git clone https://github.com/pegasilab/rassine
cd rassine

# create a virtual environment for RASSINE
python -m venv .venv

poetry install
# poetry install --all-extras # if you want the Sphinx stuff

# Download spectra and unpack them
# this creates the spectra_library subfolder
./init_spectra_library

# run ./init_spectra_library anytime you want to restart with clean data

poetry run ./run_rassine.sh -l INFO -c harpn.ini spectra_library/HD110315/data/s1d/HARPN
poetry run ./run_rassine.sh -l INFO -c harps03.ini spectra_library/HD23249/data/s1d/HARPS03


# to build the documentation if you installed with "--all-extras"
poetry run make -C docs html
``` 