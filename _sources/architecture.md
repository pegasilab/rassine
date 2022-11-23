# Architecture of RASSINE

There are three layers in the RASSINE code.

- The base layer is the Python package [rassine](https://github.com/pegasilab/rassine/tree/master/src/rassine).
- The middle layer is composed of [command-line tools](cli/index.rst) that are quite generic in the paths/filenames
  they process.
- The top and user-facing layer is composed of the [run_rassine.sh](https://github.com/pegasilab/rassine/blob/master/run_rassine.sh)
  script, which assumes a [default](pipeline.md) naming scheme for folders and files.

## Base layer

The Python code should be relatively modern; inputs and outputs are typed. The data is stored
in pickles, with the hierarchy provided by Python dicts. Those dicts are documented using TypedDict.

The `rassine` package is subdivised in submodules according to the processing step. Each step
has a `data.py` file that contains the datatype definitions.

The base layer is documented in the [API section](api.rst) of the website.

## Middle layer

RASSINE is split into [command-line tools](cli/index.rst). They [configpile](https://github.com/denisrosset/configpile)
to read and process command-line parameters, while keeping documentation about them.

Communication between those tools is done using pickled data files and CSV text files.

The middle layer is documented in the [CLI section](cli/index.rst) of the website.

## Top layer

[run_rassine.sh](https://github.com/pegasilab/rassine/blob/master/run_rassine.sh) is a Bash script
that ties all the steps together. It is opinionated about the structure of the filesystem, so that
it is compatible notably with YARARA.

Processing is done in parallel using [GNU Parallel](https://www.gnu.org/software/parallel/) which
should run in Linux and macOS.

When a step is run in paralllel, it will often write information about the processing into a CSV file.
The script makes sure that this CSV is empty before the processing, so that the programs running
concurrently can append rows to that CSV file in parallel. The file is locked before being written
to as to avoid race conditions. Of course, the resulting file does not have its rows in the
proper order, so a RASSINE script is run to sort the file after the processing step is done.

The top layer is described in the [quick start guide](quickstart.md).

## Testing

Testing of RASSINE is barebones; we have two "light" datasets that were created by removing most
of the observations from two stars:

- [HD110315_light](https://github.com/pegasilab/HD110315_light
- [HD23249_light](https://github.com/pegasilab/HD23249_light)

The tests are written using the [bats](https://bats-core.readthedocs.io/) framework (simple Bash
scripts). Those tests are run in the Github CI to check that everything is OK.

See the [dev docs](dev/tests.md) for more information.

## Documentation

The documentation is written using Sphinx and a few addons.

See the [dev docs](dev/documentation.md) for more information.
