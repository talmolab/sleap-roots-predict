---
description: Adversarial multi-lens PR review — subagent team posts a structured verdict to GitHub
---

# PR Code Review — Subagent Team

You are a senior engineer reviewing a pull request for `talmolab/sleap-roots-predict`. You
value correctness, test discipline, downstream-artifact compatibility, and maintainability
above all else.

This command launches **5 specialized subagents in parallel** to critically review the PR.
Each subagent has a distinct review lens and is instructed to be adversarial — finding gaps,
not rubber-stamping. After all subagents return, synthesize findings into a unified review
and act based on the mode determined in Step 1.

**Arguments:** `$ARGUMENTS` (optional PR number; if omitted, reviews the current branch)

## Step 0: Ground the Review in the Domain

Before preparing prompts, (re)read `openspec/project.md` and `README.md` so the lenses
match this repo: a **SLEAP prediction service** that runs `sleap-nn` inference on root
images / timelapse sequences and emits artifacts consumed by downstream `sleap-roots`
trait tooling. The five lenses below are already tailored to that domain.

## Step 1: Determine Mode

**Mode A — PR number provided** (`$ARGUMENTS` is a number):
- Gather PR context from GitHub and post a review verdict.

**Mode B — No PR / branch provided** (`$ARGUMENTS` is empty or a branch name):
- Compare against the merge base: `git diff $(git merge-base HEAD main)..HEAD`
- Report findings only; do not post to GitHub.

Resolve the repo for GitHub calls:

```bash
gh repo view --json nameWithOwner -q .nameWithOwner
```

## Step 2: Gather Context

Run in parallel:

```bash
# Mode A only — PR metadata
gh pr view $PR_NUMBER --json title,body,baseRefName,headRefName,author,labels,files

# Mode A only — full diff
gh pr diff $PR_NUMBER

# Mode A only — CI status
gh pr checks $PR_NUMBER

# Mode A only — existing automated review comments
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh api graphql -f query='
query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviews(first: 10) {
        nodes {
          author { login }
          comments(first: 50) { nodes { path line body } }
        }
      }
    }
  }
}
' -F owner="${REPO%%/*}" -F name="${REPO##*/}" -F pr=$PR_NUMBER \
  --jq '.data.repository.pullRequest.reviews.nodes[].comments.nodes[] | "File: \(.path):\(.line)\n\(.body)"'

# Mode B only — branch diff against merge base
git diff $(git merge-base HEAD main)..HEAD
```

Also read any OpenSpec proposal linked in the PR body (look for `openspec/changes/` paths).

## Step 3: Launch Subagent Review Team

Launch ALL 5 subagents **in a single message** (parallel execution). Embed the full diff,
PR description, CI status, and any automated review comments in each prompt. Each subagent
MUST read the actual source files it needs using Read/Grep — do not rely on summaries.
Each returns findings in BLOCKING / IMPORTANT / SUGGESTION tiers plus an overall score
(1–10) with justification.

```
Subagent 1: Code Quality & Architecture
  - Naming, magic numbers, module boundaries between predict.py / video_utils.py /
    plates_timelapse_experiment.py, error handling, dead code, google-style docstrings
    (ruff D), ripple effects in files the PR did not touch, public API exposed in __init__.

Subagent 2: Testing & TDD Discipline
  - Red-green-refactor followed? New code covered by tests in tests/? GPU-only paths
    correctly marked @pytest.mark.gpu so CPU CI stays green? Edge-case fixtures used
    (RGBA / greyscale / large / unicode-path)? Will tests pass on ubuntu/windows/mac
    without a GPU and without network access? Do existing tests still pass?

Subagent 3: Inference & Numerical Correctness
  - sleap-nn predictor construction + device selection correct? Array dtypes/shapes and
    channel handling right (e.g. greyscale RGB weights, RGBA)? Natural-sort preserves
    temporal order? H5 round-trips losslessly? No silent precision loss in image
    conversion. Numeric assertions use tolerances, not exact float equality.

Subagent 4: Artifact & Downstream Compatibility / Data Integrity
  - Output artifacts (labels/.slp, metadata CSV, H5) stay in the format expected by
    downstream sleap-roots consumers. Metadata provenance (timestamp, plate number)
    parsed correctly from filenames. No breaking change to file layout/column names
    without a documented migration. Paths are cross-platform (pathlib, not string concat).

Subagent 5: Behavioural Correctness & Edge Cases
  - Does the implementation match the PR description / linked spec? Trace call chains
    end-to-end (find_image_directories → load_images → make_video → predict). Adversarial
    inputs: empty directories, malformed filenames, mixed suffixes, zero-length images,
    duplicate timestamps. Existing automated review comments addressed?
```

## Step 4: Synthesize and Act

After ALL subagents return:

1. **Deduplicate** overlapping findings.
2. **Prioritize**:
   - **BLOCKING** — must fix before merge (broken tests, wrong inference output, artifact-format break, spec mismatch)
   - **IMPORTANT** — should fix before merge (missing edge cases, cross-platform risk)
   - **SUGGESTION** — optional improvements
3. **Determine verdict**: `APPROVE` / `REQUEST_CHANGES` / `COMMENT`.

**Mode A — post review to GitHub:**

> GitHub does not allow requesting changes or approving your own PRs.
> Always attempt the desired action first; if it fails with "Can not request changes on your
> own pull request" or "Can not approve your own pull request", automatically fall back to
> `--comment` with the same body and a note at the top indicating the intended verdict.

```bash
BODY="$(cat <<'EOF'
## Review Summary

[2-3 sentence overall assessment]

## Blocking Issues

[Must fix before merge — or "None"]

## Important Issues

[Should fix before merge — or "None"]

## Suggestions

[Optional improvements — or "None"]

---
*Review by Claude Code subagent team (Code Quality | Testing | Inference Correctness | Artifact Compatibility | Edge Cases)*
EOF
)"

# REQUEST_CHANGES (fall back to --comment on own-PR error):
gh pr review $PR_NUMBER --request-changes -b "$BODY" 2>&1 || \
  gh pr review $PR_NUMBER --comment -b "$(printf '> **Verdict: REQUEST_CHANGES** (posted as comment — GitHub does not allow requesting changes on your own PR)\n\n%s' "$BODY")"

# APPROVE (fall back to --comment on own-PR error):
gh pr review $PR_NUMBER --approve -b "$BODY" 2>&1 || \
  gh pr review $PR_NUMBER --comment -b "$(printf '> **Verdict: APPROVE** (posted as comment — GitHub does not allow approving your own PR)\n\n%s' "$BODY")"

# COMMENT (no fallback needed):
gh pr review $PR_NUMBER --comment -b "$BODY"
```

**Mode B — report only:**

Print the synthesized review. Do not call `gh pr review`.

5. Show the user the full synthesized review. In Mode A, also show the GitHub link.

## Related commands

- `/lint`, `/test` — run checks locally before reviewing
- `/pre-merge` — full pre-merge gate
- `/review-openspec` — review the spec before reviewing the implementation PR
