# Fusion Knowledge Base

`knowledgebase/` is the public, human-auditable source of truth for Fusion's optimization knowledge.

Fusion compiles these Markdown files into:

- the searchable SQLite BM25 index at `knowledge/knowledge.db`

That keeps the shipped CLI fast and offline-capable while keeping the underlying knowledge transparent.

## Layout

- `sources/`: external references, docs, repos, papers, benchmark posts
- `gpus/`: GPU profiles and capability notes
- `strategies/`: ranked decision cards for what to try
- `skills/`: executable playbooks for the agent
- `examples/`: known-good kernels, runtime patches, and patterns
- `documents/`: long-form notes, research summaries, and kernel writeups

## File Format

Each document uses YAML front matter followed by Markdown body content.

Example:

```md
---
id: blackwell_attention_cutile
kind: strategy
title: Blackwell Attention With cuTile
category: attention
summary: Prefer cuTile-based attention tuning on Blackwell before generic kernels.
support_level: recommended
gpu_families:
  - Blackwell
workloads:
  - decode
  - prefill
operators:
  - attention
precision:
  - fp8
source_ids:
  - nvidia-cuda-tile-flash-attention
---

## Actions

- Establish a baseline.
- Profile the current attention path.
- Try cuTile first on Blackwell.
```

## Updating The KB

For a private local knowledge base:

```bash
fusion update kb
```

This bootstraps `~/.config/fusion/knowledgebase/` if it does not exist yet and rebuilds the SQLite index in `~/.config/fusion/knowledge/`. Future Fusion runs will prefer that local rebuilt knowledge base over the embedded default.

For repo/public updates:

```bash
make kb
```

That rebuilds `knowledge/` from `knowledgebase/` and refreshes the committed SQLite index for releases.
