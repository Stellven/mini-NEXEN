---
name: apply_skill_hints
description: Apply user-provided skill hints to activate skills before trigger checks.
inputs: skill_hints
outputs: active_skills, skill_guidance
---

Purpose: Allow users to explicitly activate skills listed in the query artifact.

Steps:
- Read skill_hints from the query artifact.
- Activate matching skills and inject their guidance.

Notes:
- This runs before trigger-based skills.
