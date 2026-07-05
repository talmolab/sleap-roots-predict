---
description: Full pre-merge gate — format, lint, test, build — with PR creation, CI monitoring, and review triage.
---

Run all quality checks, create or update the PR, and prepare for merge.

## Phase 1: Code Quality

1. **Format check.** Run `/fix-formatting` first if you know there are formatting issues; otherwise run the check directly:
   ```bash
   uv run black --check sleap_roots_predict tests
   ```
   If it fails, run `/fix-formatting` and re-check.

2. **Lint.** (this repo has no type checker — ruff covers docstrings, codespell covers spelling):
   ```bash
   uv run ruff check sleap_roots_predict/
   uv run codespell
   ```
   Fix all errors before proceeding.

## Phase 2: Tests

3. **Tests.**
   - **CPU subset** (as the CPU CI runners do):
     ```bash
     uv run pytest -m "not gpu" tests/
     ```
   - **GPU subset — REQUIRED, run locally.** CI no longer runs GPU tests (the self-hosted
     GPU runner was retired — see `ci.yml`), so the `gpu`-marked tests MUST be run locally on
     a CUDA/MPS machine before merge. Install the hardware extra for your platform, then run:
     ```bash
     uv sync --extra dev --extra windows_cuda   # or linux_cuda / macos
     uv run pytest -m gpu tests/
     ```
     They must **pass** (not skip). If this machine has no accelerator, GPU cannot be verified
     here — hand it to someone with a CUDA/MPS box and record the result before merging.
   Run `/coverage` for a detailed report if coverage is a concern. Fix any failures first.

## Phase 3: Build

4. **Build artifacts** (only if packaging or the image changed):
   ```bash
   uv build                                   # PyPI wheel
   docker build -t sleap-roots-predict .      # GHCR service image
   ```
   Fix any build errors before opening a PR.

## Phase 4: Documentation

5. **Docs review.** Run `/docs-review` if the change touches public-facing docs, the
   public API (`__init__.py`), or output artifact formats.
   - Check that `README.md` and `CLAUDE.md` reflect the change.
   - Remove any stale references to deprecated code.

## Phase 5: OpenSpec Verification

6. **Check proposal status** (if this branch has an active OpenSpec change):
   ```bash
   openspec list
   ```
   - Verify `openspec/changes/<change-id>/tasks.md` — all items should be `- [x]`.
   - Run `openspec validate <change-id> --strict` and fix any issues.
   - After merge, archive via `/cleanup-merged` (uses the CLI — never a manual `git mv`).

## Phase 6: Pull Request

7. **Create or update the PR.**
   - Run `/pr-description` to generate the PR body.
   - Push all changes:
     ```bash
     git push -u origin $(git branch --show-current)
     ```
   - Create the PR if it doesn't exist yet:
     ```bash
     gh pr create --base main --head $(git branch --show-current) \
       --title "feat: <descriptive title>" \
       --body "$(cat <<'EOF'
     ## Summary
     ...
     EOF
     )"
     ```

## Phase 7: CI Monitoring

8. **Monitor GitHub Actions.** After pushing, watch CI:
   ```bash
   gh pr checks $(gh pr list --head $(git branch --show-current) --json number -q '.[0].number')
   ```
   - If any job fails, run `/ci-debug` to investigate.
   - View failed logs: `gh run view <RUN_ID> --log-failed`.
   - Fix incrementally and push.

## Phase 8: Review Feedback

9. **Triage Copilot comments.** Run `/copilot-review` — address high-priority issues immediately, evaluate medium-priority suggestions, document any you disagree with.

10. **Triage PR comments.** Run `/review-pr` and address all concerns from human reviewers and automated checks.

11. **Plan fixes for complex issues.** Use `superpowers:brainstorming` for issues that need design thought. Implement incrementally, push, and re-watch CI.

## Phase 9: Changelog

12. **Update changelog.** Run `/update-changelog` if this repo tracks one. Follow semantic versioning (MAJOR breaking / MINOR feature / PATCH fix).

## Phase 10: Final Verification

13. **Final gate.** Confirm:
    ```bash
    git fetch origin main
    git merge-base --is-ancestor origin/main HEAD
    ```
    No merge conflicts; all CI checks green; all review comments addressed; the **Phase 2 GPU
    subset passed locally** (CI does not cover GPU). Run one last combined check:
    ```bash
    uv run black --check sleap_roots_predict tests && \
      uv run ruff check sleap_roots_predict/ && \
      uv run codespell && \
      uv run pytest -m "not gpu" tests/
    ```

## Output Format

```markdown
## Pre-Merge Check Results

- [x/!/ ] Format: PASS / pre-existing issue (#N) / FAIL
- [x/!/ ] Lint (ruff + codespell): PASS / ...
- [x/!/ ] Tests (CPU): X passed
- [x/!/ ] Tests (GPU, local): X passed / N/A (no accelerator — verified elsewhere)
- [x/!/ ] Build/Image: PASS / N/A
- [x/!/ ] Docs: current / updated
- [x/!/ ] OpenSpec: all tasks complete / N/A
- [x/!/ ] PR: #N created/updated; CI green
- [x/!/ ] Copilot: no blockers / X addressed
- [x/!/ ] Changelog: updated / N/A

Status: READY TO MERGE
```

Merge command:
```bash
gh pr merge <PR_NUMBER> --squash --delete-branch
```

Post-merge: run `/cleanup-merged`.

## When to Skip Phases

Document your reasoning when skipping a phase:

- **Docs-only PR** — skip build/test/coverage; run docs-review.
- **Hotfix** — skip coverage and docs; run format + lint + test.
- **Config or dependency bump** — still build the image if deps changed (inference stack).

## Related Commands

- `/fix-formatting` — auto-format before the check
- `/lint` — ruff + codespell + format check
- `/coverage` — test coverage report
- `/copilot-review` — triage GitHub Copilot inline comments
- `/review-pr` — adversarial multi-lens PR review
- `/ci-debug` — debug a failing CI run
- `/update-changelog` — maintain the changelog
- `/pr-description` — generate the PR body
- `/cleanup-merged` — post-merge branch cleanup + OpenSpec archive
