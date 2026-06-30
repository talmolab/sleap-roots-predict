---
description: Maintain CHANGELOG.md following Keep a Changelog format with SemVer
---

# Update Changelog

Maintain CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

> Note: this repo does not yet have a `CHANGELOG.md`. Create one from the template below the
> first time you run this; the version lives in `sleap_roots_predict/__init__.py`
> (`__version__`), surfaced via the `[tool.setuptools.dynamic]` attr in `pyproject.toml`.

## Quick Commands

```bash
# View recent commits
git log --oneline --decorate -10

# View commits since last tag
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# Detailed, reverse-chronological (easier for writing entries)
git log $(git describe --tags --abbrev=0)..HEAD --pretty=format:"%h %s" --reverse

# Current version
uv run python -c "import sleap_roots_predict; print(sleap_roots_predict.__version__)"
```

## Changelog Format

CHANGELOG.md follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles:
changelogs are for humans, latest version first, one section per release, `YYYY-MM-DD` dates,
[SemVer](https://semver.org/) version numbers.

### Change Categories

- **Added** — New features
- **Changed** — Changes to existing functionality
- **Deprecated** — Soon-to-be removed features
- **Removed** — Removed features
- **Fixed** — Bug fixes
- **Security** — Security vulnerability fixes

## Changelog Template

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New feature description

### Fixed

- Bug fix description

[Unreleased]: https://github.com/talmolab/sleap-roots-predict/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/talmolab/sleap-roots-predict/releases/tag/v0.0.1
```

## Workflow: Adding Changes

### Step 1: Identify Changes Since Last Release

```bash
git tag -l | sort -V | tail -1
git log <last-tag>..HEAD --pretty=format:"%h %s" --reverse
```

### Step 2: Categorize Each Change

Group commits by category (Added / Changed / Fixed / Security / Removed / Deprecated).
Skip: CI config changes, test-only refactors, minor internal churn (unless user-facing).

### Step 3: Update `[Unreleased]`

```markdown
## [Unreleased]

### Added

- Feature name with brief description (#issue-number)

### Fixed

- Bug description — what broke and what the correct behaviour is (#issue-number)
```

### Step 4: When Releasing a Version

Move `[Unreleased]` to a versioned section, bump `__version__` in
`sleap_roots_predict/__init__.py`, and update the comparison links:

```markdown
[Unreleased]: https://github.com/talmolab/sleap-roots-predict/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/talmolab/sleap-roots-predict/compare/v0.0.1...v0.1.0
```

## Writing Good Changelog Entries

### Good

```markdown
### Added

- `batch_predict()` — run inference across multiple H5 files in one call (#45)

### Fixed

- Greyscale conversion used wrong channel weights for RGBA inputs (#42)
```

### Bad

```markdown
- New stuff               ← too vague
- Fix bug                 ← wrong category; use Fixed
- Updated dependencies    ← skip unless breaking change
```

## Breaking Changes

```markdown
### Changed

- **BREAKING**: `process_timelapse_experiment()` now requires an explicit `models` argument.
  - Migration: pass `models=None` to preserve the previous metadata-only behaviour.
```

## Tips

1. Update `[Unreleased]` continuously as PRs merge — do not batch at release time
2. Link to issues/PRs `(#42)` for traceability
3. Write for users ("Added batch prediction") not implementation jargon
4. Mark breaking changes with `**BREAKING:**` and a migration path
5. Skip internal-only changes
6. Keep dates accurate (`YYYY-MM-DD`)

## Release Checklist

- [ ] All `[Unreleased]` entries moved to the new versioned section
- [ ] `__version__` bumped per SemVer (`MAJOR.MINOR.PATCH`)
- [ ] Date is today's date in `YYYY-MM-DD`
- [ ] Comparison links updated
- [ ] Breaking changes marked with `**BREAKING:**` and migration notes
- [ ] No duplicate section headers; no placeholder dates

## Related Commands

- `/review-pr` — PR review includes a documentation / changelog check
- `/pre-merge` — full pre-merge gate before merging
- `/openspec:archive` — archive the OpenSpec change after the PR merges
