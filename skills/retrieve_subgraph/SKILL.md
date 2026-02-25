---
name: retrieve_subgraph
description: Expand the document set by retrieving a KG subgraph and evidence.
inputs: topic, interests, user_profile
outputs: documents
---

Purpose: Use the knowledge graph to expand retrieval beyond lexical matches.

Steps:
- Seed entities from the query and user profile.
- Retrieve a small subgraph and associated evidence.
- Add evidence documents to the document set.
