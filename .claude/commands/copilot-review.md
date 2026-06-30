---
description: Pull and triage all GitHub Copilot inline review comments on the current branch's PR.
---

Find and display all GitHub Copilot comments on the current branch's pull request, then offer to address them.

## Step 1: Find the PR

```bash
BRANCH=$(git branch --show-current)
gh pr list --state open --head "$BRANCH"
```

If no PR exists, inform the user and exit — create one with `/pr-description` first.

## Step 2: Get PR Number

Extract the PR number from the list output.

## Step 3: Fetch Copilot Review via GraphQL

Use the GraphQL API — it is the only reliable method for retrieving Copilot review comments:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh api graphql -f query='
query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviews(first: 10) {
        nodes {
          author { login }
          body
          comments(first: 50) {
            nodes { path body diffHunk }
          }
        }
      }
    }
  }
}' -F owner="${REPO%%/*}" -F name="${REPO##*/}" -F pr=<PR_NUMBER>
```

## Step 4: Parse Copilot Comments

From the response, extract:

- Reviews where `author.login == "copilot-pull-request-reviewer"`
- `body` — the overview, which may contain a "Comments suppressed due to low confidence" note
- `comments.nodes` — file-specific inline suggestions, each with `path`, `body`, and `diffHunk`

## Step 5: Categorize

1. **High Priority** — bugs, wrong inference output, type/shape errors
2. **Medium Priority** — code quality, maintainability, best practices
3. **Low Priority / Informational** — style suggestions, low-confidence notes

## Step 6: Display Formatted Summary

```markdown
# GitHub Copilot Review for PR #<N>

**Branch**: <branch-name>
**PR Title**: <title>
**Repo**: talmolab/sleap-roots-predict

## Overview

[Copilot's general PR overview comment]

## High Priority Issues (<count>)

1. **File**: path/to/file:42
   - **Issue**: Description of the problem
   - **Suggestion**: What Copilot recommends
   - **Confidence**: High / Medium / Low
   - **Status**: Open / Fixed in <commit>

## Medium Priority Suggestions (<count>)

[Same format]

## Low Priority / Informational (<count>)

[Same format]

## Summary

- Total comments: <N>
- High priority: <N>
- Medium priority: <N>
- Low priority / informational: <N>

## Recommended Actions

- [Specific tasks to address the feedback]
```

## Step 7: Offer to Address

After displaying the summary, ask the user:

```
Would you like me to:
1. Address all high-priority issues now
2. Create a plan to address specific issues
3. Explain any of these suggestions in detail
4. Mark low-confidence suggestions as reviewed (document why they're being skipped)
```

## Edge Cases

- **No Copilot comments** — report "No GitHub Copilot comments found on this PR."
- **PR not found** — suggest creating a PR via `/pr-description` first.
- **Multiple open PRs for this branch** — list all and ask which to check.
- **Copilot not enabled** — inform the user that Copilot reviews aren't enabled for this repo.

## Best Practices

- Always run this before requesting human review.
- Address high-confidence suggestions promptly.
- Evaluate low-confidence suggestions carefully — they may be false positives.
- When ignoring a suggestion, document why in the PR or a code comment.

## Related Commands

- `/review-pr` — adversarial multi-lens PR review (includes a Copilot check pass)
- `/pre-merge` — full pre-merge gate (includes this command)
- `/ci-debug` — debug CI failures that Copilot may have flagged
