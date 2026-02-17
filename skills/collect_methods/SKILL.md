---
name: collect_methods
description: Load recorded analysis methods from the personal library.
inputs: topic
outputs: methods
---

Purpose: Ensure the agent has access to user-defined analysis methods (lenses).

Steps:
- Read the methods table in the local SQLite database.
- Attach the most recent methods to the working context.

Notes:
- Used at the start of each research round.
