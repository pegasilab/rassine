How to release a new RASSINE version
====================================

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
git tag "$CUR_VERSION" -a -m "Version $CUR_VERSION"
```
