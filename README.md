# RASSINE

RASSINE is a tool to ..


## Test framework

```
git clone https://github.com/pegasilab/rassine
cd rassine

# create a virtual environment for RASSINE
python -m venv .venv

poetry install

git submodule update --init test/bats
git submodule update --init test/test_helper/bats-assert
git submodule update --init test/test_helper/bats-support

poetry run test/bats/bin/bats test
```

## Outdated

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

poetry run ./run_rassine.sh -l INFO -c harpn.ini spectra_library/HD110315/data/s1d/HARPN
poetry run ./run_rassine.sh -l INFO -c harps03.ini spectra_library/HD23249/data/s1d/HARPS03


# to build the documentation if you installed with "--all-extras"
poetry run make -C docs html
``` 

