# How to release a new RASSINE version

All the following commands need to be run in the base folder of the `rassine` repository.

## Release commit and tag

We first create a commit that describes the canonical version that will be published later. There is a commit and an associated tag, so that the particular release will be referenced in 

Be on the master branch, in the base folder, and check that all changes have already be committed.

```bash
git checkout master
git status
```

Get the current version number.

```bash
poetry version
```

The version number uses the PEP 440 convention `major.minor.patch`.

Decide whether you want to want to bump the `major`, `minor` or `patch` part.

```bash
PREV_VERSION=$(poetry version --short)
poetry version minor # change accordingly
CUR_VERSION=$(poetry version --short)
git add pyproject.toml
git commit -m "Bumping $PREV_VERSION -> $CUR_VERSION"
git tag "v$CUR_VERSION" -a -m "Version $CUR_VERSION"
```

If the previous lines ran without error, run the following to push this information to GitHub. If something went wrong, restart from the beginning on a freshly cloned repository.

```bash
git push origin master
git push origin v$CUR_VERSION
```

## Publish the package on PyPI (username/password)

Run the following command to publish this version on PyPI.

```bash
poetry publish --build --username [YOUR PYPI USERNAME]
```

## Create the release on GitHub (optional)

Go to [https://github.com/pegasilab/rassine/tags](https://github.com/pegasilab/rassine/tags), click on the latest tag, and select "Create release from tag". Fill in the details.
