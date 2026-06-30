---
description: End-to-end workflow for scoping, proposing, and implementing a new feature using superpowers brainstorming and TDD.
---

You are starting a new feature workflow. The user's feature request is: $ARGUMENTS

This repo uses **two complementary planning systems** — they layer, they don't compete:

- **superpowers** (`brainstorming`, `writing-plans`, `subagent-driven-development`, `test-driven-development`) — drive the conversational design, planning, and implementation discipline
- **OpenSpec** (`openspec/`, the `openspec` CLI) — produces durable spec deltas for any change that adds capabilities, modifies behavior, or affects architecture

For non-trivial features, use BOTH. For tiny changes (typo, formatting, dependency bump, test for existing behavior), skip OpenSpec but still follow superpowers TDD discipline.

> **Scope guard:** the sleap-nn inference rebuild + warm-GPU worker is roadmap tier
> **A3-predict** — do not undertake it as an incidental feature. See `openspec/project.md`.

## Guardrails

- Do NOT write any implementation code until the OpenSpec proposal is approved by the user.
- Follow OpenSpec conventions strictly — see `openspec/AGENTS.md` for the authoritative rules.
- Use TDD when implementing — write failing tests before implementation code.
- Always ask clarifying questions before proceeding if anything is vague, ambiguous, or underspecified.

## Steps

1. **Ensure feature branch.** Check the current branch (`git branch --show-current`). If on `main`, ask the user what branch name to create — suggest a kebab-case, verb-led name based on the feature (e.g., `add-cli-entrypoint`, `fix-greyscale-weights`). Create and switch to it before proceeding.

2. **Invoke `superpowers:brainstorming`.** This is mandatory even for changes that seem simple. The brainstorming skill explores user intent, requirements, and design through clarifying questions, then produces a design doc at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`. Do not skip.

3. **Explore the codebase.** Use subagents (Explore agent type) to understand current state relevant to this feature. Investigate:
   - Existing related code in `sleap_roots_predict/`
   - Existing OpenSpec specs: `openspec spec list --long`
   - Existing OpenSpec changes: `openspec list`
   - Tests that exercise the affected area in `tests/`

4. **Decide OpenSpec scope.** Based on brainstorming + exploration, decide:
   - Is this a **new capability** (new spec) or a **modification to an existing capability** (delta on existing spec)?
   - What are the affected capabilities? (Each affected capability gets its own delta.)
   - Pick a unique, kebab-case, verb-led `change-id`.

5. **Create the OpenSpec proposal.** Invoke `/openspec:proposal` with the change-id and grounding context from steps 2–3. The proposal scaffolds:
   - `openspec/changes/<change-id>/proposal.md` — what and why
   - `openspec/changes/<change-id>/tasks.md` — ordered, verifiable work items. Tasks MUST explicitly outline a TDD approach: for each task, specify what tests will be written first and what behavior they verify.
   - `openspec/changes/<change-id>/design.md` — only if the solution spans multiple systems, introduces a new pattern, or has trade-offs worth documenting
   - `openspec/changes/<change-id>/specs/<capability>/spec.md` — one folder per affected capability, using `## ADDED|MODIFIED|REMOVED Requirements` with at least one `#### Scenario:` per requirement

6. **Validate strictly.** Run `openspec validate <change-id> --strict` and fix every issue before sharing the proposal.

7. **Get user approval.** Present the validated proposal to the user and wait for explicit approval before proceeding to implementation. Surface:
   - The change-id and one-line summary
   - The list of affected capabilities and their delta types (ADDED / MODIFIED / REMOVED)
   - Any open questions or trade-offs from `design.md`

8. **Implement with TDD.** Once approved, invoke `superpowers:writing-plans` to create the implementation plan, then `superpowers:subagent-driven-development` (or implement directly for smaller changes). For each task:
   - Write the failing test first (`superpowers:test-driven-development` skill)
   - Implement the minimum code to pass
   - Mark the task complete (`- [x]`) in `tasks.md`
   - Run `/lint` and `/test` before moving to the next task

9. **Pre-merge sweep.** Before opening a PR, run `/pre-merge` (format check + lint + test + build).

10. **Open a PR.** Use `/pr-description` for the template. Reference the OpenSpec change-id in the description.

11. **After merge: clean up** on `main`. See `/cleanup-merged`. Verify all `tasks.md` items are `- [x]` first.

## Reference

- **superpowers skills**: invoke via the `Skill` tool; `using-superpowers` describes the meta-process
- **Project context**: `CLAUDE.md` at repo root
- **OpenSpec rules**: `openspec/AGENTS.md` (canonical) and `openspec/project.md` (this project's stack and conventions)
- **OpenSpec sub-commands**: `/openspec:proposal`, `/openspec:apply`, `/openspec:archive` (auto-generated by `openspec init`)

## Related Commands

- `/openspec:proposal` — scaffold the OpenSpec proposal (step 5)
- `/openspec:apply` — implement an approved proposal (alternative to step 8)
- `/openspec:archive` — archive after merge (called from `/cleanup-merged`)
- `/test` — run tests during TDD cycle
- `/lint` — lint during implementation
- `/pre-merge` — final gate before opening PR
- `/pr-description` — generate the PR body
- `/cleanup-merged` — post-merge cleanup
