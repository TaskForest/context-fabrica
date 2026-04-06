# Releasing to PyPI

`context-fabrica` publishes to PyPI through GitHub Actions trusted publishing.

## Pre-release checklist

```bash
rm -rf dist build src/context_fabrica.egg-info
python -m build
python -m twine check dist/*
python -m pytest
python -m pip install .
```

Optional but recommended:

```bash
python -m pip install ".[all]"
python -m venv /tmp/context-fabrica-release-test
. /tmp/context-fabrica-release-test/bin/activate
python -m pip install dist/*.whl
context-fabrica --help
context-fabrica-doctor --help
```

## Versioning

1. Update `setup.py` version.
2. Add the release section to `CHANGELOG.md`.
3. Commit and push to `main`.

## Publish

1. Create a GitHub release tagged like `v1.0.0`.
2. The `publish-pypi.yml` workflow will build, run `twine check`, and publish to PyPI.
3. Approve the `pypi` GitHub environment if required.

## GitHub repository setup required once

Before the first PyPI publish, a repository admin must create the `pypi` environment in GitHub:

1. Go to `Settings -> Environments`
2. Create an environment named `pypi`
3. Optionally require reviewers for manual approval

I attempted to create it automatically, but the current GitHub token does not have repository admin rights.

## Trusted publishing

Configure PyPI trusted publishing for:

- repository: `TaskForest/context-fabrica`
- workflow: `.github/workflows/publish-pypi.yml`
- environment: `pypi`

## After publish

Verify the released package:

```bash
python -m pip install context-fabrica
context-fabrica --help
```
