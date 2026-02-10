---
name: collect_interests
description: Load recorded user interests from the personal library.
inputs: topic
outputs: interests
---

Purpose: Ensure the agent has access to the user's historical interests.

Steps:
- Read the interests table in the local SQLite database.
- Attach the most recent interests to the working context.

Notes:
- Used at the start of each research round.
