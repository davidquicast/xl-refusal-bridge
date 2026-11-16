# XL-Refusal-Bridge

**Where does multilingual safety live in Tiny Aya, and can a two-vector adapter move it?**

Lexsi AI Safety and Alignment Hackathon — Track 2 (multilingual safety, locating and steering safety behaviour).
Model under test: [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35 B parameters, 36 layers).

## The question

A multilingual model has to do two things before it refuses a harmful request: **recognise** that the request is
harmful, and **route** that recognition into a refusal. We test whether these two steps degrade differently across
languages:

- **H1 — detection is language-invariant.** A single linear probe at a middle layer separates harmful from benign
  prompts equally well in all ten languages, including a language it never saw in training.
- **H2 — routing is not.** The projection of harmful prompts onto the English refusal direction shrinks as the
  language gets lower-resource, and that shrinkage predicts unsafe behaviour.
- **H3 — a bridge closes the gap.** Using the invariant detector to gate a projection clamp along the refusal
  direction restores refusals where routing is weak, at a small cost in over-refusal and none in capability.

Every hypothesis is stated so that it can fail, and the results section reports which parts held.

## Method

All activations are read at the last prompt token, on the residual stream after each block.

| Step | What is computed | Data it may see |
|---|---|---|
| Refusal direction | `r_L = mean(h_L | harmful, en) − mean(h_L | benign, en)`, unit-normalised | English train |
| Detector | LDA probe with a language-invariance penalty: `w = (S_W + β·S_lang + λ·tr(S_W)/d·I)⁻¹ (μ_H − μ_B)` | train, 10 languages; XSTest-safe prompts are hard negatives |
| Probe selection | layer `L_d`, `β`, `λ` maximising the **worst-language** validation AUROC | validation |
| Gate | `g = 1[w·h_{L_d} > τ]`, `τ` = 95th percentile of safe validation scores (benign + XSTest-safe) | validation |
| Bridge | `h ← h + g · max(0, c·m*_L − r̂_L·h) · r̂_L` at layer(s) `L_r`, on the last prompt token and every generated token | `m*` from English train |
| Tuning | `L_r`, `c` with the lowest validation ASR subject to benign over-refusal ≤ baseline + 4 pp | validation, 4 languages |

When the gate is closed the forward pass is bit-identical to the base model. The adapter is one probe vector, one
direction per steered layer and two scalars; nothing is fine-tuned.

### Benchmark

`xl-refusal-bench`: the same prompts in ten languages (en, zh, it, vi, ar, ko, th, bn, sw, jv).

- **Harmful** — MultiJail (315 prompts, human translations).
- **Benign** — Alpaca instructions without input, machine-translated with NLLB-200 (1.3 B).
- **Safe but alarming** — XSTest safe prompts, same translation pipeline.

A machine-translated prompt is kept only if its back-translation reaches chrF ≥ 25 and it is in the right script in
*every* language; otherwise it is dropped everywhere, so all languages stay parallel. Splits are made by prompt, never
by row, so no prompt appears on both sides of a split in any language.

### Measurement

Responses are translated to English and labelled by a rubric judge (Qwen2.5-7B-Instruct) with two booleans, *refusal*
and *harmful*. A second model (PolyGuard-Qwen) gives an independent refusal label, and the baseline is also judged in
the original language where the judge supports it; agreement is reported.

- **ASR** — share of harmful prompts that get a harmful response.
- **RR** — share of harmful prompts refused.
- **ORR** — share of benign / XSTest-safe prompts refused.
- **Gap** — ASR of the worst language minus English ASR, and the low-resource-tier mean minus English.
- **Capability** — Belebele reading-comprehension accuracy, first-token KL on safe prompts, tokens per second.

Before/after differences use an exact McNemar test per language with Holm correction; rates carry bootstrap 95 %
intervals.

## Lexsi kits in the chain

| Kit | Role |
|---|---|
| **CuratorKIT** | cleaning and near-duplicate removal of the benchmark, dataset card |
| **SafeTune** | two steering baselines (`RefusalDirectionTrainer`, `CASTTrainer`) and an independent re-derivation of the refusal direction |
| **CircuitKIT** | EAP-IG circuit discovery for refusal in English, Swahili and Bengali, and their overlap |
| **AuditKIT** | Belebele scoring with a significance test, and a red-team probe set before/after |

<!-- RESULTS -->

## Reproduce

```bash
pip install -e .
python tests/dry_run.py            # offline end-to-end check on a tiny random model, no GPU
```

The full study runs as two GPU sessions that coordinate through a Hugging Face dataset repository (`notebooks/`).
Session A builds the benchmark, caches activations, fits and tunes the bridge and evaluates it; session B runs the
kit stages. Each stage is idempotent and writes a `run.json` with code revision, config hash, seed, hardware and
library versions.

```bash
python -m xlrb.cli run_a           # bench → acts → fit → base → tune → bridge → ablations → report
python -m xlrb.cli run_b           # SafeTune baselines → capability → red team → circuits
python -m xlrb.cli ood             # held-out prompts and unseen languages
python -m xlrb.figures results/<tag> results/<tag> article/figures
```

## Layout

```
src/xlrb/      pipeline (data, activations, fit, bridge, judge, metrics, kits, cli, figures)
configs/       the one configuration used for the reported run
notebooks/     the two Colab entry points
results/       metrics, per-language table, analysis and kit outputs
adapters/      the bridge (safetensors + config)
article/       write-up and figures
space/         Gradio demo
tests/         offline dry run
```

## Limitations

- One model family and one harmful-prompt source; MultiJail prompts are direct requests, not adversarial jailbreaks.
- Benign and XSTest prompts are machine-translated; quality is lowest exactly where the question is most interesting.
- Verdicts come from an LLM judge reading machine-translated responses. Agreement figures are reported, but there is
  no human annotation.
- The refusal direction is a difference of means: it shows that a direction is sufficient to steer, not that it is
  the mechanism the model uses.

## License

MIT for the code. The benchmark inherits the licences of MultiJail, Alpaca and XSTest.
