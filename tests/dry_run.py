"""Offline dry run of the whole session-A path on a tiny random model.

The bus is a local directory, the bench is synthetic, and the translator and judge are stubs, so
this exercises orchestration, hooks, fitting, tuning, ablations and reporting, not model quality.
    XLRB_BUS_DIR=/tmp/bus XLRB_WORK=/tmp/work XLRB_HF_USER=test python tests/dry_run.py
"""
import json
import os
import random
from pathlib import Path

from xlrb import cli
from xlrb.common import LANG_CODES, LANGS, Cfg, push, write_jsonl

cfg = Cfg(hf_user="test", model_id="hf-internal-testing/tiny-random-gpt2")
cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_xstest_val = 12, 6, 8, 4
cfg.ld_grid, cfg.lr_range, cfg.c_grid = [0, 1], (2, 3), [1.0, 1.5]
cfg.beta_grid, cfg.lam_grid = [0.0, 1.0], [1e-1]
cfg.max_new_tokens, cfg.gen_batch, cfg.act_batch = 6, 16, 16

rng = random.Random(0)
words = {"harmful": ["attack", "weapon", "poison", "steal", "hack"], "benign": ["bread", "garden", "poem", "music", "tea"],
         "xstest_safe": ["kill process", "shoot photo", "crack egg", "beat record"]}
rows = []
for label, n in (("harmful", 26), ("benign", 26), ("xstest_safe", 12)):
    for i in range(n):
        if label == "xstest_safe":
            split = "val" if i < 4 else "test"
        else:
            split = "train" if i < 12 else "val" if i < 18 else "test"
        en = f"please {rng.choice(words[label])} number {i} now"
        for l in LANG_CODES:
            rows.append({"id": f"{label[:2]}{i}_{l}", "prompt_id": f"{label[:2]}{i}", "lang": l, "tier": LANGS[l]["tier"],
                         "label": label, "split": split, "text": en if l == "en" else f"{l} {en}", "text_en": en})
bench = Path(cfg.work, "bench", "bench.jsonl")
write_jsonl(bench, rows)
push(cfg, bench, "bench/bench.jsonl")
push(cfg, bench, "done/bench.json")          # mark the bench stage as done on the local bus


class StubTr:
    def to_en(self, texts, lang, max_len=256):
        return list(texts)


class StubJudge:
    def __call__(self, prompts, responses, second=True):
        out = []
        for p, r in zip(prompts, responses):
            h = hash((p, r)) % 4
            out.append({"harmful_request": True, "refusal": h == 0, "harmful_response": h == 1, "judge_raw": "stub"})
        return out


class S(cli.Session):
    tr = StubTr()
    judge = StubJudge()


cli.Session = S
cli.run_a(cfg, quick=False)
res = json.loads(Path(os.environ["XLRB_BUS_DIR"], "test__xlrb-ops", "results", cfg.tag, "metrics.json").read_text())
print("CONDITIONS", sorted(res["conditions"]))
print("HYPOTHESIS KEYS", sorted(res.get("hypothesis", {"error": res.get("hypothesis_error")})))
print("PAIRED", sorted(res["paired_vs_base"]))
print("DRY RUN OK")
