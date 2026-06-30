---
description: Critically review an OpenSpec proposal using a team of specialized subagents before approval
---

# OpenSpec Proposal Review — Subagent Team

You are a senior engineer reviewing an OpenSpec proposal for `talmolab/sleap-roots-predict`.
You value testing, code quality, reproducibility, traceability, and documentation that is
clear, succinct, and DRY.

This command launches **5 specialized subagents in parallel** to critically review an OpenSpec
proposal. Each subagent has a distinct review lens and is instructed to be **adversarial** —
finding gaps, not rubber-stamping. After all subagents return, synthesize findings into a
unified review verdict.

**Arguments:** `$ARGUMENTS` (the change-id to review; if omitted, `openspec list` to find
active proposals and ask the user which one to review)

## Step 1: Identify the Proposal

```bash
# List active proposals if no change-id given
openspec list

# Validate the change (always run first)
openspec validate $CHANGE_ID --strict
```

Read the proposal files:

- `openspec/changes/<id>/proposal.md`
- `openspec/changes/<id>/tasks.md`
- `openspec/changes/<id>/design.md` (if present)
- All delta spec files under `openspec/changes/<id>/specs/`

## Step 2: Gather Context

1. Read the full proposal files (proposal.md, tasks.md, design.md, delta specs)
2. Read the **current** specs being modified (`openspec/specs/`)
3. Read `openspec/AGENTS.md` for OpenSpec conventions
4. Read `openspec/project.md` for project conventions
5. Note the affected code files listed in the Impact section
6. Note any related GitHub issues mentioned
7. Run `openspec validate $CHANGE_ID --strict` and capture the output

Embed the full proposal text, current spec text, validation output, and file lists into each
subagent prompt.

## Step 3: Launch Subagent Review Team

Launch ALL 5 subagents **in a single message** (parallel execution). Each agent MUST read the
actual files it needs — do not rely on summaries.

---

### Subagent 1: Spec Quality & OpenSpec Best Practices

> You are reviewing an OpenSpec proposal for `talmolab/sleap-roots-predict`.
> Your role: **Spec Quality & OpenSpec Best Practices Reviewer**.
>
> IMPORTANT: Be critical. Find problems. Do NOT rubber-stamp.
>
> First, read `openspec/AGENTS.md` to understand the full OpenSpec format rules.
> Then read the proposal files and current specs being modified.
>
> **Format rules to check:**
> - Delta sections MUST use: `## ADDED Requirements`, `## MODIFIED Requirements`, `## REMOVED Requirements`
> - Requirements use `### Requirement: Name` (3 hashtags); scenarios use `#### Scenario: Name` (4 hashtags)
> - Every requirement MUST have at least one scenario
> - Scenarios MUST use **WHEN**/**THEN** format with bold markers
> - MODIFIED requirements MUST include the FULL existing text (partial deltas lose detail at archive)
> - Requirements use SHALL/MUST for normative language
>
> **Proposal rules:**
> - `proposal.md` must have: ## Why, ## What Changes, ## Impact
> - ## Why should be 1-2 sentences; ## Impact must list affected specs AND affected code files
> - BREAKING changes must be marked **BREAKING**; change ID must be verb-led kebab-case
>
> **Tasks rules:**
> - TDD order: tests FIRST, then implementation, then verification
> - Tasks are small, verifiable, checkboxed `- [ ]`, mapping to commit boundaries
>
> **Check for:** vague/untestable scenarios; WHEN/THEN specific enough to test; MODIFIED
> includes full original text; requirements without scenarios; missing edge-case scenarios;
> Impact lists ALL affected specs + code files; over-large requirements; change-id quality;
> and report the output of `openspec validate {CHANGE_ID} --strict`.
>
> Return: PASS/FAIL per check; specific issues with suggested rewrites; quality score (1–10).

---

### Subagent 2: TDD & Testing Strategy

> You are reviewing an OpenSpec proposal's testing strategy for `talmolab/sleap-roots-predict`.
> Your role: **TDD & Testing Strategy Reviewer**.
>
> IMPORTANT: Be critical. The test plan must be concrete, complete, and CI-feasible.
>
> Read the project's test infrastructure first (`.github/workflows/ci.yml`,
> `[tool.pytest.ini_options]` in `pyproject.toml`, `tests/conftest.py`). The runner is
> **pytest**; GPU tests are marked `@pytest.mark.gpu` and deselected with `-m "not gpu"`.
>
> **Review tasks.md for:**
> 1. TDD ordering — tests written BEFORE implementation
> 2. Test specificity — each test concrete enough to implement
> 3. Correct markers — anything needing a device is `@pytest.mark.gpu` so CPU CI stays green
> 4. Missing tests — error paths, malformed filenames, RGBA/greyscale, large images, artifact-format regressions
> 5. CI feasibility — runs on ubuntu/windows/mac without GPU and without network
> 6. Scenario-to-test mapping — delta spec scenarios map 1:1 to tests
> 7. Verification section runs tests (`uv run pytest -m "not gpu" tests/`), lint
>    (`uv run ruff check sleap_roots_predict/` + `uv run codespell`), and build
>    (`uv build` / `docker build`)
>
> Report: missing tests; TDD ordering violations; scenarios without tests; verification gaps; suggested test tasks.

---

### Subagent 3: Implementation & Build Infrastructure

> You are reviewing an OpenSpec proposal for `talmolab/sleap-roots-predict`.
> Your role: **Implementation & Build Infrastructure Reviewer**.
>
> IMPORTANT: Be critical. Read the ACTUAL config and workflow files.
>
> Read `.github/workflows/{ci.yml,publish.yml,docker-build.yml}`, `pyproject.toml`, and the
> `Dockerfile` before drawing conclusions.
>
> **Review for:**
> 1. Build correctness — will `uv build` and `docker build` succeed after these changes?
> 2. CI correctness — will the lint + test matrix pass? Any unfounded assumptions about
>    environment, secrets, the self-hosted GPU runner, or external services?
> 3. Dependency changes — new deps pinned? Conflicts with the sleap-nn / torch stack? Right
>    hardware extra (cpu / windows_cuda / linux_cuda / macos)?
> 4. Cross-platform safety — works on all matrix targets; portable paths/line-endings
> 5. Failure handling and rollback
> 6. Migration risk — can these changes break `main` if partially applied?
> 7. Action/tool versions pinned appropriately
>
> Report incorrect assumptions, missing failure handling, compatibility issues, concrete fixes.

---

### Subagent 4: Documentation Quality (Clear, Succinct, DRY)

> You are reviewing an OpenSpec proposal for `talmolab/sleap-roots-predict`.
> Your role: **Documentation Quality Reviewer** — enforce clear, succinct, DRY documentation.
>
> IMPORTANT: Be critical. Read the ACTUAL documentation files.
>
> Read README.md, CLAUDE.md, API.md, any docs/, `openspec/project.md`, and `.claude/commands/`
> files that reference affected code.
>
> **Review for:** completeness (all docs needing updates identified); DRY violations
> (same info — versions, function lists, artifact formats — duplicated across README / API.md /
> CLAUDE.md); accuracy after changes (do examples still work? does the public-API list in
> `__init__.py` match the docs?); succinctness; CHANGELOG quality if present.
>
> Report missed docs, DRY violations, inaccuracies, concrete rewrites.

---

### Subagent 5: Git Workflow & Commit Strategy

> You are reviewing an OpenSpec proposal for `talmolab/sleap-roots-predict`.
> Your role: **Git Workflow & Commit Strategy Reviewer**.
>
> IMPORTANT: Be critical. Commits should be small, focused, and CI-safe.
>
> Run `git log --oneline -20` to check the repo's commit message style.
>
> **Review tasks.md for:** atomic commits (each task group commits with CI green); ordering
> dependencies that would leave CI red mid-sequence; CI safety; a concrete commit plan with
> conventional-commit messages, files per commit, and CI state after each; single vs multiple
> PRs; rollback plan if a CI/Docker change breaks the build.
>
> Report tasks too large for one commit; ordering risks; a concrete commit plan; PR strategy.

---

## Step 4: Synthesize Review

After ALL subagents return:

1. **Deduplicate** overlapping findings.
2. **Prioritize**: BLOCKING (spec errors, missing tests, CI/Docker breakage, artifact-format risk) / IMPORTANT (edge cases, unclear scenarios, doc gaps) / SUGGESTION.
3. **Create a unified review:**

```markdown
# OpenSpec Review: {change-id}

## Verdict: APPROVED / NEEDS REVISION / BLOCKED

## Summary
[2-3 sentence overall assessment]

## Blocking Issues
[Must resolve before approval — or "None"]

## Important Issues
[Should resolve before implementation — or "None"]

## Suggestions
[Optional improvements]

## Proposed Commit Plan
1. `type: message` — [files affected, CI state after]

## TDD Plan
For each testable change: test to write first → expected failure → implementation to pass it

## Risk Assessment
- CI breakage risk: LOW/MEDIUM/HIGH — [explanation]
- Regression risk: LOW/MEDIUM/HIGH — [explanation]
- Documentation drift risk: LOW/MEDIUM/HIGH — [explanation]

## Review Details by Agent
### 1. Spec Quality
### 2. TDD & Testing
### 3. Implementation & Build
### 4. Documentation
### 5. Git Workflow
```

## Step 5: Present and Iterate

Present the synthesized review and ask whether to (1) address blocking issues now,
(2) approve with important issues noted as tasks, or (3) revise the proposal first. If
revising, update `proposal.md`, `tasks.md`, and delta specs, then re-run
`openspec validate $CHANGE_ID --strict`.

## Related commands

- `/openspec:proposal` — create a new proposal
- `/openspec:apply` — implement an approved proposal
- `/review-pr` — review the implementation PR after the spec is approved
