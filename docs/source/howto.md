How to
======

All the following commands need to be run in the base folder of the `rassine` repository.

Run tests
---------

Before releasing RASSINE, if any untested changes were made, verify that all tests pass.

```bash
poetry run test/bats/bin/bats test
```

Build the documentation locally
-------------------------------

It is good practice to verify that the documentation builds successfully.

To build the documentation, you additionally need the [Graphviz](https://graphviz.org/download/) tool.
It is provided by standard package managers on Linux. On macOS, it needs to be installed either using
[MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/).

You need 

```bash
poetry run make -C docs clean html
```

then open the built website.

```bash
xdg-open docs/build/html/index.html # Linux
open docs/build/html/index.htmlxdg-open docs/build/html/index.html # macOS
```
