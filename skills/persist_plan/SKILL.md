---
name: persist_plan
description: Render the plan and outline into markdown for storage.
inputs: plan, outline
outputs: plan_md
---

Purpose: Convert the plan to a markdown file.

Steps:
- Render the plan, outline, and interests into markdown.
- The supervisor handles file naming and saving.

Notes:
- Output is used by the backend for downstream workflows.
