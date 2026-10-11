# Releasing snn_opt

Results produced with this package record `snn_opt.__version__` as the solver
identity, so a version string must name exactly one body of code.
`tests/test_version.py` enforces the rule below: an untagged checkout must carry
a development version.

## Versions

- `main` always carries the next development version, `X.Y.Z.devN` (for
  example `0.9.0.dev0` after `0.8.1`). A checkout of `main` therefore never
  reports a released version.
- A released version `X.Y.Z` appears only on the commit its `vX.Y.Z` tag points
  at.
- The version lives in `pyproject.toml` and `src/snn_opt/__init__.py`, which
  must agree. `CITATION.cff` names the latest *release* and changes only when a
  release is cut.

## Minor or major release (from `main`)

1. Move the `[Unreleased]` entries of `CHANGELOG.md` under `## [X.Y.0] - date`.
2. Set the version to `X.Y.0` in both files and in `CITATION.cff`
   (`version`, `date-released`); run the test suite; commit.
3. Tag `vX.Y.0` on that commit and push the tag. The `wheels` workflow builds
   every platform and publishes to PyPI on a `v*` tag push.
4. Set `main` to the next development version (`X.(Y+1).0.dev0`), commit, push.

## Patch release (when `main` already holds unreleased work)

A patch must not ship `main`'s unreleased features, so it is built from the
released code:

1. Branch `release/X.Y.x` from the tag `vX.Y.Z` (keep it local; the public
   remote accepts only `main` and release tags reachable from `main`).
2. Commit the fix with its regression test, then the release commit: version
   `X.Y.(Z+1)` in both files, `CITATION.cff`, and a `CHANGELOG.md` entry.
   Tag `vX.Y.(Z+1)`.
3. Merge `release/X.Y.x` into `main`. **Resolve the version files to `main`'s
   development version** (keep `X.(Y+1).0.devN`; take the release's
   `CITATION.cff` and `CHANGELOG.md` entry). `tests/test_version.py` fails if
   the merge leaves the release version on an untagged `main` commit.
4. Push `main`, then push the tag (it is now reachable from `main`).
