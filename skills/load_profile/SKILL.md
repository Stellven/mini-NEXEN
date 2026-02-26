---
name: load_profile
description: Load profile signals from the local KG profile table.
inputs: topic
outputs: extracted_interests
---

Purpose: Ensure planning and retrieval can use stored profile signals.

Steps:
- Read the kg_user_profile table and select top interest terms.
- Attach profile terms to the working context.

Notes:
- Profiles are rebuilt from local documents via kg-build-local.
