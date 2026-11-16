"""Judge validation: score each rubric variant of the primary judge against reference labels.

The reference file (`judge/reference.json` on the bus) holds prompt/response pairs with labels from a
much larger model (`ref_refusal`, `ref_harmful`). Output: agreement and Cohen's kappa per variant and
per prompt class, plus the raw verdicts, in `judge/validation.json`.
"""
from __future__ import annotations

import json

import numpy as np

from .common import Cfg, pull, push
from .judge import RUBRIC_SYSTEM, RUBRIC_VARIANTS, _ChatClassifier, parse_rubric, rubric_prompt


def kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe + 1e-12)


def score(rows: list[dict], name: str) -> dict:
    out = {}
    for key, ref in (("refusal", "ref_refusal"), ("harmful", "ref_harmful")):
        for lab in ("all", "harmful", "benign", "xstest_safe"):
            rs = [r for r in rows if (lab == "all" or r["label"] == lab)
                  and r.get(ref) is not None and r.get(f"{name}_{key}") is not None]
            if rs:
                a, b = [r[f"{name}_{key}"] for r in rs], [r[ref] for r in rs]
                out[f"{key}/{lab}"] = {"n": len(rs), "rate": float(np.mean(a)), "ref_rate": float(np.mean(b)),
                                       "agreement": float(np.mean(np.array(a) == np.array(b))),
                                       "kappa": kappa(a, b)}
    return out


def main(cfg: Cfg | None = None):
    cfg = cfg or Cfg()
    rows = json.loads(open(pull(cfg, "judge/reference.json")).read())
    clf = _ChatClassifier(cfg.judge_id, cfg.judge_batch)
    res = {"judge_id": cfg.judge_id, "n": len(rows), "variants": {}}
    for name in RUBRIC_VARIANTS:
        raw = clf.run(RUBRIC_SYSTEM, [rubric_prompt(r["text_en"], r["response_en"], name) for r in rows], 24)
        for r, t in zip(rows, raw):
            v = parse_rubric(t)
            r[f"{name}_refusal"], r[f"{name}_harmful"] = v["refusal"], v["harmful_response"]
        res["variants"][name] = score(rows, name)
        print(f"[judgecheck] {name}: {json.dumps(res['variants'][name])[:600]}")
    res["rows"] = rows
    out = cfg.wdir("judge", "validation.json")
    out.write_text(json.dumps(res, ensure_ascii=False))
    push(cfg, out, "judge/validation.json")


if __name__ == "__main__":
    main()
