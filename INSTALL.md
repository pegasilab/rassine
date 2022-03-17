
# Installation instructions

- install poetry

- clone the RASSINE project

- git submodule update --init to obtain the test spectra library

- in the RASSINE cloned project, run "python -m venv .venv" to create a dedicated virtual environment

- run `poetry install`

- if GNU Parallel is not installed (for example on macOS), copy the `parallel` and `sem` scripts
  to the .venv/bin folder
   
- execute `poetry run ./run_harps03.sh` in the base folder
