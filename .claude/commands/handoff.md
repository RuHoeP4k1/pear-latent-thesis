---
description: Update HANDOFF.md at the end of a session so the next session (on any machine) can continue
allowed-tools: Read, Edit, Bash(git status:*), Bash(git log:*), Bash(git diff:*)
---

## Context

Recent commits:
!`git log --oneline -10`

Uncommitted changes:
!`git status --short`

## Your task

Update `HANDOFF.md`:

1. Rewrite the "Current state" section: date (today), phase, what was done in this session, the
   single next step (concrete enough to start without asking), and what is blocking.
2. Remove answered items from "Open questions" and add new ones. An answer only counts if it
   comes from a Tier 1 source or from Ruben.
3. Update the "Unverified assumptions in code" table: add any new `# UNVERIFIED:` markers
   (search the code for them), remove ones that were verified.
4. Add a session log entry at the top of "Session log": three to six lines, decisions first.
5. Plain literal language, no metaphors.

Then list uncommitted changes and propose a commit message. Do not commit or push unless Ruben
says so. $ARGUMENTS
