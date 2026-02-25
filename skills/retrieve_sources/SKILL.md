---
name: retrieve_sources
description: Retrieve relevant documents from the personal library based on the topic and interests.
inputs: topic, interests
outputs: documents
---

Purpose: Identify the most relevant documents for the current research topic.

Steps:
- Load documents from the library.
- Score them using keyword overlap with the topic and interests.
- Return the combined documents.

Notes:
- If no documents exist, leave the list empty.
