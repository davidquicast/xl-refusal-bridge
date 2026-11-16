### Aggregate (test split)

| Condition | ASR macro ↓ | Refusal on harmful ↑ | Over-refusal benign ↓ | Over-refusal XSTest ↓ | Gap max (pp) ↓ | Gap low-tier (pp) ↓ | Languages |
|---|---|---|---|---|---|---|---|
| Tiny Aya (base) | 5.4 | 69.9 | 6.0 | 17.6 | 9.3 | 7.8 | 10 |
| + gated bridge (ours) | 0.1 | 95.6 | 8.2 | 24.4 | 0.7 | 0.2 | 10 |
| SafeTune refusal direction | 1.7 | 77.9 | 6.0 | 22.4 | 4.0 | 2.9 | 10 |
| SafeTune CAST | 0.9 | 98.2 | 99.6 | 98.1 | 8.0 | 0.0 | 10 |

### Ablations (same prompts for every row: 5 languages, test split)

| Condition | ASR macro ↓ | Refusal on harmful ↑ | Over-refusal benign ↓ | Over-refusal XSTest ↓ | Gap max (pp) ↓ | Gap low-tier (pp) ↓ | Languages |
|---|---|---|---|---|---|---|---|
| Tiny Aya (base) — same subset | 6.4 | 68.2 | 7.0 | 17.4 | 9.0 | 7.7 | 5 |
| + gated bridge (ours) — same subset | 0.4 | 96.4 | 9.0 | 26.5 | 1.0 | 0.3 | 5 |
| ablation: additive | 2.0 | 91.6 | 7.6 | 22.1 | 4.0 | 3.0 | 5 |
| ablation: no gate | 0.4 | 97.4 | 87.0 | 93.5 | 1.0 | 0.7 | 5 |
| ablation: random direction | 7.6 | 76.6 | 6.8 | 18.4 | 10.0 | 8.7 | 5 |
| ablation: English-only probe | 2.0 | 91.2 | 7.2 | 21.2 | 4.0 | 3.3 | 5 |
| ablation: beta = 0 | 0.8 | 96.2 | 7.8 | 24.2 | 2.0 | 1.0 | 5 |
| ablation: no hard negatives in the probe | 2.0 | 92.2 | 6.8 | 61.2 | 5.0 | 3.0 | 5 |

### Per language: base → bridge (test split)

| Lang | Tier | ASR % (95% CI) base | ASR % (95% CI) bridge | Holm p (McNemar) | Refusal % | Over-refusal benign % | Over-refusal XSTest % | Gate on harmful % |
|---|---|---|---|---|---|---|---|---|
| en | ref | 0.7 (0.0–2.0) | 0.0 (0.0–0.0) | 1 | 69.3 → 93.3 | 0.7 → 2.7 | 9.3 → 15.1 | 98.7 |
| zh | high | 4.7 (1.3–8.0) | 0.0 (0.0–0.0) | 0.0938 | 68.0 → 98.7 | 4.7 → 7.3 | 15.1 → 19.8 | 99.3 |
| it | high | 4.7 (2.0–8.0) | 0.0 (0.0–0.0) | 0.0938 | 60.0 → 88.0 | 6.0 → 6.0 | 12.8 → 20.9 | 97.3 |
| vi | high | 4.7 (1.3–8.0) | 0.0 (0.0–0.0) | 0.0938 | 78.7 → 97.3 | 9.3 → 10.0 | 19.8 → 24.4 | 99.3 |
| ar | mid | 4.7 (2.0–8.0) | 0.0 (0.0–0.0) | 0.0938 | 68.7 → 94.0 | 9.3 → 9.3 | 19.8 → 24.4 | 98.0 |
| ko | mid | 5.3 (2.0–9.3) | 0.0 (0.0–0.0) | 0.0547 | 72.7 → 96.0 | 3.3 → 3.3 | 20.9 → 22.1 | 99.3 |
| th | mid | 4.0 (1.3–7.3) | 0.7 (0.0–2.0) | 0.125 | 77.3 → 96.0 | 12.7 → 15.3 | 29.1 → 38.4 | 95.3 |
| bn | low | 8.7 (4.7–13.3) | 0.7 (0.0–2.0) | 0.00439 | 64.7 → 95.3 | 6.0 → 9.3 | 11.6 → 22.1 | 98.7 |
| sw | low | 10.0 (6.0–14.7) | 0.0 (0.0–0.0) | 0.00061 | 66.0 → 98.7 | 3.3 → 8.0 | 19.8 → 29.1 | 98.0 |
| jv | low | 6.7 (3.3–11.3) | 0.0 (0.0–0.0) | 0.0156 | 74.0 → 98.7 | 4.7 → 10.7 | 17.4 → 27.9 | 98.7 |