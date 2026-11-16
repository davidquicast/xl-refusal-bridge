---
license: mit
base_model: CohereLabs/tiny-aya-global
language: [en, zh, it, vi, ar, ko, th, bn, sw, jv]
tags: [safety, refusal, activation-steering, multilingual, lexsi-hackathon]
---
# Tiny Aya XL-Refusal-Bridge

An inference-time adapter for [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global):
a language-invariant linear probe at layer 20 gates a projection clamp along the English refusal direction at
layers 25, 27 and 29. No weights are changed. When the gate is closed the forward pass is identical to the base model.

| File | Content |
|---|---|
| `bridge.safetensors` | `w` (probe, 2,048 floats) and `r.25`, `r.27`, `r.29` (unit refusal directions) |
| `config.json` | layers, threshold `tau`, strength `c`, target projections `m_star`, selection metadata |

## Results (test split, ten languages, 150 harmful prompts per language)

| | Base | + bridge |
|---|---|---|
| Attack success (macro) | 5.4 % | 0.1 % |
| Refusal of harmful prompts | 69.9 % | 95.6 % |
| Refusal of benign prompts | 6.0 % | 8.2 % |
| Refusal of XSTest-safe prompts | 17.6 % | 24.4 % |
| Worst-language gap to English (attack success) | 9.3 pp | 0.7 pp |
| Belebele accuracy | 75.6 % | 75.6 % |

Single harmful verdicts of the automatic judge are noisy; the refusal rate is the better-measured outcome. Over-refusal on alarming-but-safe prompts rises by about 7 pp. See the article in the code
repository for method, ablations, transfer to unseen languages and limitations.

All 25,220 test responses were re-labelled by a ten-times-larger judge (Qwen2.5-72B, same rubric): attack
success 7.5 % → 0.1 % (significant in nine of ten languages), refusal 77.3 % → 96.6 %, over-refusal on benign
prompts 4.1 % → 5.8 % and on XSTest-safe prompts 26.7 % → 32.2 %.

## Use

```python
from huggingface_hub import snapshot_download
from xlrb.activations import generate
from xlrb.bridge import Bridge, load_bridge
from xlrb.common import load_lm

spec, w, r = load_bridge(snapshot_download("davidquicast/tiny-aya-xl-refusal-bridge"))
model, tok = load_lm(spec.model_id)
print(generate(model, tok, ["Ninawezaje kuoka mkate nyumbani?"], 96, 1, ctx=Bridge(model, spec, w, r)))
```

`xlrb` is the package in the code repository, https://github.com/davidquicast/xl-refusal-bridge (`pip install -e .`).
Benchmark: https://huggingface.co/datasets/davidquicast/xl-refusal-bench · demo: https://huggingface.co/spaces/davidquicast/xl-refusal-bridge

## Limits

Fitted and evaluated on direct harmful requests (MultiJail), not on adversarial jailbreaks. One model. The refusal
direction is a difference of means: sufficient to steer, not a claim about mechanism.
