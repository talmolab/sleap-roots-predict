---
description: Clean up a merged branch and archive its OpenSpec change via the OpenSpec CLI
---

Clean up a feature branch after its PR merges. Archive any completed OpenSpec proposal
**with the OpenSpec CLI — never by hand-moving folders** (the CLI dates the archive folder
AND promotes the change's delta specs into the live `openspec/specs/` baseline, which a
manual `git mv` skips).

## Step 1: Confirm the PR is actually merged

Never delete a branch before GitHub confirms the PR merged.

```bash
gh pr view <pr-number> --json state,mergedAt,headRefName
```

Proceed **only if** `"state": "MERGED"`. If it is `OPEN` or `CLOSED` (not merged), stop —
do not delete the branch.

## Step 2: Switch to main and pull

```bash
git checkout main
git pull origin main
```

## Step 3: Delete the local feature branch

```bash
git branch -d <branch-name>
```

On a **squash merge**, git prints `warning: the branch '<branch>' is not yet merged to
HEAD` because the squashed commit is not an ancestor of `main`. **This is expected and the
delete still succeeds.** Only fall back to force-delete when Step 1 already confirmed
`MERGED`:

```bash
# Only after gh confirmed state == MERGED:
git branch -D <branch-name>
```

## Step 4: Prune the stale remote-tracking ref

```bash
git fetch --prune origin
```

If the remote branch was not auto-deleted (repo setting off):
`git push origin --delete <branch-name>`.

## Step 5: Archive the OpenSpec change with the CLI

Be on `main` (with the merged PR pulled) before archiving — archiving on a feature branch
will not update the base specs on `main`.

```bash
# 1. Find the active change id
openspec list

# 2. Confirm all tasks are complete (never archive with incomplete tasks)
grep -c '\- \[ \]' openspec/changes/<id>/tasks.md   # expect 0

# 3. Validate the change before archiving
openspec validate <id> --strict

# 4. Archive — dates the folder (archive/YYYY-MM-DD-<id>/), promotes the change's
#    delta specs into openspec/specs/, and re-validates
openspec archive <id> --yes
```

- Use `--skip-specs` **only** for changes with no spec deltas at all (pure tooling/docs).
- When several changes touch the same capability, archive in dependency order
  (base/parent change first).

## Step 6: Verify the archive and promoted specs

```bash
openspec spec list --long          # the promoted / updated specs
openspec validate --all --strict   # everything still valid
git status openspec/               # the archive rename + new specs/ files
```

## Final step: Commit and push

Use `git add -A` so BOTH the archive rename AND the newly-promoted `openspec/specs/` files
are captured. A partial `git add openspec/changes` would silently drop the promoted spec files.

```bash
git add -A
git commit -m "chore: clean up after PR #<n> (archive <id>)"
git push origin main
```

## Fallback ONLY when the OpenSpec CLI is unavailable

If `openspec` genuinely cannot run, archive by hand — but treat this as a **documented
fallback that needs review**, because you must replicate what the CLI does:

```bash
# 1. Date the folder yourself
git mv openspec/changes/<id> "openspec/changes/archive/$(date +%F)-<id>"
# 2. HAND-PROMOTE each delta from that change's specs/ into openspec/specs/
#    (the step most often missed — "specs are truth" depends on it)
# 3. Validate
openspec validate --all --strict
```

Flag the commit/PR as **"manual OpenSpec archive — needs review"**.

## Post-cleanup checklist

- [ ] `gh pr view <n> --json state` showed `MERGED` before any deletion
- [ ] Local branch gone; stale remote ref pruned (`git fetch --prune origin`)
- [ ] `openspec list` no longer shows the change; `openspec/specs/` updated (if it had deltas); archive folder dated `archive/YYYY-MM-DD-<id>/`; `git add -A` captured both rename and specs

## Related commands

- `/openspec:archive` — the underlying archive step this wraps
- `/pr-description`, `/review-pr`, `/pre-merge`
