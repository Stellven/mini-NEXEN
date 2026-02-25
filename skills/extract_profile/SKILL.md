---
name: extract_profile
description: Extract user profile signals (interest, intent, focus, attention) from documents.
inputs: documents
outputs: user_profile
---

Purpose: Build a user profile from local documents to personalize research outputs.

Steps:
- Use the LLM to infer interest, intent, focus, and attention items.
- Store profile edges with salience and optional time ranges.
- Fall back to keyword frequency from local docs if LLM extraction is unavailable.
