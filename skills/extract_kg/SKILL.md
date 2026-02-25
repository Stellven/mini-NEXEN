---
name: extract_kg
description: Extract KG triples and claims from selected documents.
inputs: documents
outputs: kg_entities, kg_relations, kg_claims, kg_evidence
---

Purpose: Build the local knowledge graph from the current document set.

Steps:
- Load the selected documents.
- Use the LLM to extract triples and claims.
- Store entities, relations, claims, mentions, and evidence.

Notes:
- Skips documents that were already extracted.
