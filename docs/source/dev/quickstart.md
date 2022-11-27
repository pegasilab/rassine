# Quickstart (development version)

To develop RASSINE, you have first to install the [Poetry](https://python-poetry.org/) tool
to manage dependencies and the RASSINE virtual environment.

Poetry may or may not play well with Conda.

After having installed Poetry, verify (`python --version`) that you have at least Python 3.8
installed.

## Prepare the working directory and the dependencies

First, clone the RASSINE repository.

```bash
git clone git@github.com:pegasilab/rassine.git
cd rassine
```

Then, download the Git submodules. Those modules contain the [bats](https://bats-core.readthedocs.io/)
testing framework.

We skip downloading the other submodules containing spectral data files for now (they can be big).

```bash
git submodule update --init test/bats
git submodule update --init test/test_helper/bats-assert
git submodule update --init test/test_helper/bats-support
```

Then, create a virtual environment so that your manipulations do not affect other Python
environments in your system. The `.venv` directory in the base folder is a convention recognized
by `Poetry` and `Visual Studio Code`.

```bash
python -m venv .venv
```

Then install RASSINE and its dependencies.

```bash
poetry install -E docs
```

You can remove the `-E docs` parameter if you do not plan to build the documentation locally
using Sphinx.

If you do plan to build the documentation, you will need to install the [Graphviz](https://graphviz.org/download/)
command-line tool. Unfortunately, it is not part of `PyPI`.

For example, on Ubuntu:

```bash
sudo apt install graphviz
```

On macOS, you will need to install [Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/),
and then install graphviz.

## Verify that the tests run

We run the tests, and prefix the `bats` command with `poetry run`. Running commands with the
`poetry run` prefix makes sure that the command runs in the virtual environment managed by Poetry
(in our case, the one in `.venv/`).

```bash
poetry run test/bats/bin/bats test
```

## Verify that the documentation builds

```bash
poetry run make -C docs clean html
```

and open the `docs/build/html/index.html` file.


## Run full reductions


```bash
git submodule update --init spectra/HD23249
git submodule update --init spectra/HD110315

poetry run ./run_rassine.sh -l INFO -c harpn.ini spectra/HD110315/data/s1d/HARPN
poetry run ./run_rassine.sh -l INFO -c harps03.ini spectra/HD23249/data/s1d/HARPS03
```

## How to contribute

Pull requests are preferred: create a branch for your modification, push it to this repository
and then use the GitHub interface to merge, after all tests have passed.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests