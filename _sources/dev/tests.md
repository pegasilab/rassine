# Testing

Because the top layer of RASSINE is a Bash scripts, the tests are written
using the [bats](https://bats-core.readthedocs.io/en/stable/) framework.

We do not require people to install the `bats` framework, it is included in the RASSINE repository as a Git submodule.

The tests can be run simply using:
```bash
poetry run test/bats/bin/bats test
```

