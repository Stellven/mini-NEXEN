---
name: detect_contradictions
description: Detect contradictions between extracted claims.
inputs: kg_claims, kg_relations
outputs: kg_contradictions
---

Purpose: Flag conflicting claims to improve traceability and gap analysis.

Steps:
- Identify claim pairs that share a subject and predicate but differ on object.
- Use the LLM to validate contradictions when available.
- Store contradictions with confidence scores.
