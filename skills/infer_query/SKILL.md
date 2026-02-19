---
name: infer_query
description: Infer topic and analysis methodologies from the user query and save a reviewable artifact.
inputs: topic
outputs: inferred_topic, inferred_methods, query_artifact
---

Purpose: Infer analysis methodologies from the query to avoid manual setup.

Steps:
- Use the LLM (or heuristics fallback) to infer topic and analysis methodologies.
- Save a human-readable artifact for user review.
- Apply edited values before continuing the pipeline.

Notes:
- This skill should run before retrieval and planning steps.
