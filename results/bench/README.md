---
license: cc-by-nc-4.0
task_categories: [text-generation]
language: [en, zh, it, vi, ar, ko, th, bn, sw, jv]
tags: [safety, red-teaming, multilingual, refusal, lexsi-hackathon]
configs:
- config_name: default
  data_files: bench.jsonl
---
# XL-Refusal-Bench

Paired harmful / benign / looks-unsafe-but-safe prompts in 10 languages, for measuring
cross-lingual refusal gaps and over-refusal with the same prompt ids in every language.

| label | source | translation | prompts per language |
|---|---|---|---|
| `harmful` | MultiJail (Deng et al., 2023) | human (from MultiJail) | 315 |
| `benign` | Alpaca instructions without an input | machine, back-translation gated | 315 |
| `xstest_safe` | XSTest safe prompts (Röttger et al., 2023) | machine, back-translation gated | 206 |

Splits (`train` / `val` / `test`) are assigned by `prompt_id`, so they are identical across languages.
Machine translation: `facebook/nllb-200-distilled-1.3B`; back-translation for the gate: `facebook/nllb-200-distilled-1.3B`; a prompt is kept only if every
language passes chrF >= 25 against the English source and a script check. Prompts dropped by the
gate: {'benign': 77, 'xstest_safe': 44}. Curated with CuratorKIT (cleaning, MinHash near-duplicate removal, manifest and
checksums under `curatorkit/`): {"used": true, "cells": {"harmful": {"in": 315, "kept": 315}, "benign": {"in": 315, "kept": 315}, "xstest_safe": {"in": 206, "kept": 206}}, "dropped_prompt_ids": 0, "version": "0.1.29"}.

Fields: `id, prompt_id, lang, tier, label, split, text, text_en, source, tags, mt_system, bt_chrf`.

The harmful split contains harmful requests by design. It is intended for safety evaluation only.
