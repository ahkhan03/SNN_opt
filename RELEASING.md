# Releasing snn_opt

Results produced with this package record `snn_opt.__version__` as the solver
identity, so a version string must name exactly one body of code. The string
is derived from git. A clean commit that a `vX.Y.Z` tag points at reports
exactly `X.Y.Z`. Every other commit, and any dirty tree, reports the next
patch as `X.Y.Z.devN` plus a local segment `+g<sha>`. A dirty tree also adds
the default date marker. PyPI rejects a local version, so an untagged or
dirty build cannot be uploaded.

Nothing in the tree stores a hand-written package version. `CITATION.cff`
names the latest release and is the only version field edited by hand.
`tests/test_version.py` checks the identity: an exact tag on a clean tree
matches the tag with no local segment, any other commit carries `.dev` and
its own sha, and a git checkout's imported version matches a fresh derivation.

## Minor or major release (from `main`)

1. Move the `[Unreleased]` entries of `CHANGELOG.md` under `## [X.Y.0] - date`.
2. Set `CITATION.cff` (`version`, `date-released`) to that release. Run the
   test suite and commit.
3. Tag `vX.Y.0` on that commit and push the tag. The `wheels` workflow builds
   every platform and publishes to PyPI. The publish job accepts only a strict
   `vX.Y.Z` tag, and only after every artifact's filename, metadata, and
   packaged version equal that tag, the wheel set includes manylinux x86_64,
   manylinux aarch64, macOS arm64, and Windows AMD64, each of CPython 3.9
   through 3.14 appears at least once, and one installed wheel's runtime
   version equals that tag.
4. Push `main`. It is not the tag, so it derives a development version on its
   own. There is no version bump to commit.

## Patch release (when `main` already holds unreleased work)

A patch must not ship `main`'s unreleased features, so it is built from the
released code:

1. Branch `release/X.Y.x` from the tag `vX.Y.Z` (keep it local; the public
   remote accepts only `main` and release tags reachable from `main`).
2. Commit the fix with its regression test and a `CHANGELOG.md` entry. Set
   `CITATION.cff` to `X.Y.(Z+1)`. Tag `vX.Y.(Z+1)` on that commit.
3. Merge `release/X.Y.x` into `main` and take the release's `CITATION.cff`
   and `CHANGELOG.md` entry. There is no version file to resolve.
4. Push `main`, then push the tag (it is now reachable from `main`).
