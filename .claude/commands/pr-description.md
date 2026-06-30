---
description: Generate a comprehensive PR description from the current diff, with a three-state verification checkbox convention.
---

Use this command when opening a pull request to document what changed and what was verified.

## Quick Commands

```bash
gh pr view              # View current PR
gh pr diff              # View PR diff
gh pr diff --name-only  # List changed files
gh pr checks            # Check CI status
```

## Checkbox Convention (READ FIRST)

Use a three-state convention for verification checkboxes — don't tick `[x]` out of habit:

- `[x]` — Verified green. You ran the command and it passed.
- `[!]` — Pre-existing failure on `main`. This PR introduces no new failures. Link the issue tracking the baseline bug.
- `[ ]` — Not yet verified, or doesn't apply.

The `[!]` state exists because docs/config/refactor PRs often inherit lint/build failures
that already exist on `main`. Ticking `[x]` on those would be a false claim. Use `[!]` to be honest.

Example:
```
- [x] All tests pass — 105 passing (CPU subset)
- [!] `uv run black --check sleap_roots_predict tests` fails on pre-existing issue (#12), not introduced here
- [ ] GPU tests (not run — no GPU available locally)
```

## PR Description Template

```markdown
## Summary

[Brief 1-2 sentence description of what this PR does and why.]

## Changes

- [Bullet list of specific changes]
- [Use present tense: "Add X", "Fix Y", "Update Z"]

## OpenSpec Change

- Change ID: `<change-id>`
- Affected capabilities: `<capability-1>`, `<capability-2>`
- Delta types: ADDED / MODIFIED / REMOVED
- All `tasks.md` items complete: yes/no

(If this PR is too small for an OpenSpec proposal — typo, dep bump, test-only — say "No OpenSpec change: <reason>".)

## Testing

- [ ] CPU tests pass (`uv run pytest -m "not gpu" tests/`)
- [ ] GPU tests pass on a CUDA/MPS machine, or marked `@pytest.mark.gpu` and N/A here
- [ ] New tests added for new functions and behavior
- [ ] Existing tests still cover affected paths

## Linting

- [ ] Ruff passes (`uv run ruff check sleap_roots_predict/`)
- [ ] Codespell passes (`uv run codespell`)
- [ ] Formatting passes (`uv run black --check sleap_roots_predict tests`)

## Build / Image

- [ ] Wheel builds (`uv build`) — if packaging changed
- [ ] Docker image builds (`docker build -t sleap-roots-predict .`) — if Dockerfile / deps changed

## Downstream Compatibility

- [ ] Output artifact format (labels/.slp, metadata CSV, H5) unchanged, or migration documented
- [ ] No breaking change to the public API in `__init__.py`, or documented below

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes documented below with migration path

## Related Issues

Closes #[issue number]
Related to #[issue number]

## Examples

[If a calculation, array shape, or artifact format changed, include a worked example.]

## Reviewer Notes

[Any specific concerns, trade-offs, or areas to focus on.]
```

## GitHub CLI Tips

```bash
# Create PR with heredoc body (preferred — keeps formatting)
gh pr create --title "feat: <descriptive title>" --body "$(cat <<'EOF'
## Summary
...
EOF
)"

# Create PR with body from a file
gh pr create --title "feat: ..." --body-file pr-description.md

# Edit PR description
gh pr edit --body "Updated description"

# View failed job logs
gh run view --log-failed
```

## Related Commands

- `/pre-merge` — run the full local gate before opening a PR
- `/review-pr` — adversarial multi-lens review of this PR
- `/copilot-review` — fetch and triage GitHub Copilot inline comments
- `/cleanup-merged` — post-merge cleanup workflow
