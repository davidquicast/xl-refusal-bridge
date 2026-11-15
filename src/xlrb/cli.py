"""Stage runner. Every stage is idempotent: it is skipped when its DONE marker is on the Hub bus.

    python -m xlrb.cli bench            # build + curate + push the bench (NLLB on GPU)
    python -m xlrb.cli run_a            # Colab session A: acts -> fit -> base -> tune -> bridge -> ablations -> report
    python -m xlrb.cli run_b            # Colab session B: kit baselines, capability, circuits, red-team
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import fit as F
from .bridge import Bridge, BridgeSpec, load_bridge, save_bridge
from .common import (Cfg, is_done, load_bench, load_cfg, pull, push, read_jsonl, run,
                     wait_for, write_jsonl)

TUNE_LANGS = ["en", "bn", "sw", "jv"]
ABL_LANGS = ["en", "th", "bn", "sw", "jv"]


class Session:
    """Lazily loaded models shared by the stages of one process."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self._lm = self._tr = self._judge = None

    @property
    def lm(self):
        if self._lm is None:
            from .common import load_lm

            self._lm = load_lm(self.cfg.model_id)
        return self._lm

    @property
    def tr(self):
        if self._tr is None:
            from .judge import Translator

            self._tr = Translator(self.cfg)
        return self._tr

    @property
    def judge(self):
        if self._judge is None:
            import torch

            from .judge import Judge

            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 30 * 2**30:
                print("[warn] < 30 GB VRAM: using the small PolyGuard judge (recorded in run.json)")
                self.cfg.judge_id = "ToxicityPrompts/PolyGuard-Qwen-Smol"

            self._judge = Judge(self.cfg)
        return self._judge


def stage(cfg: Cfg, name: str, force: bool = False) -> bool:
    """True if the stage must run."""
    if not force and is_done(cfg, name):
        print(f"[skip] {name} already done")
        return False
    return True


# ----------------------------------------------------------------------------- bench (local)
def do_bench(cfg: Cfg, limit: int | None = None, public: bool = False, S=None):
    from .data import build
    from .kits import curate

    if not stage(cfg, "bench"):
        return
    with run(cfg, "bench") as rec:
        out = cfg.wdir("bench", "x").parent
        from .data import NllbMT

        stats = build(cfg, out, limit=limit, mt=NllbMT(cfg, S.tr) if S is not None else None)
        rec["metrics"]["build"] = stats
        rec["metrics"]["curatorkit"] = curate(out / "bench_raw.jsonl", out)
        # The bench lives on the private bus; the publish step copies it to the public dataset repo.
        for f in ("bench.jsonl", "bench_raw.jsonl", "build_stats.json", "README.md"):
            push(cfg, out / f, f"bench/{f}")
        if (out / "curatorkit").exists():
            push(cfg, out / "curatorkit", "bench/curatorkit")


# ----------------------------------------------------------------------------- acts + fit
def do_acts(cfg: Cfg, S: Session):
    from .activations import cache_acts

    name = f"acts_{cfg.tag}"
    if not stage(cfg, name):
        return
    with run(cfg, name) as rec:
        from .activations import selfcheck
        from .common import get_layers

        model, tok = S.lm
        nL = len(get_layers(model))
        chk = max(0, min(nL // 2, nL - 2))
        err = selfcheck(model, tok, chk)
        if err > 0.1:
            raise RuntimeError(f"hook/hidden_states mismatch at layer {chk}: max abs diff {err}")
        out = cfg.wdir("acts", cfg.tag, "x").parent
        t0 = time.time()
        rec["metrics"] = cache_acts(cfg, model, tok, load_bench(cfg), out)
        rec["metrics"].update({"n_layers": nL, "hook_vs_hidden_states_max_abs_diff": err})
        rec["metrics"]["seconds"] = round(time.time() - t0, 1)
        push(cfg, out, f"acts/{cfg.tag}")


def _acts(cfg: Cfg) -> F.Acts:
    from .activations import load_acts

    d = Path(cfg.work, "acts", cfg.tag)
    if not any(d.glob("*.npy")):
        d = pull(cfg, f"acts/{cfg.tag}")
    return F.Acts(load_acts(d), {r["id"]: r["split"] for r in load_bench(cfg)})


def do_fit(cfg: Cfg):
    name = f"fit_{cfg.tag}"
    if not stage(cfg, name):
        return
    with run(cfg, name) as rec:
        A = _acts(cfg)
        r, m_star, m0 = F.refusal_dirs(A, ["en"])
        sel = F.select_probe(A, cfg.ld_grid, cfg.beta_grid, cfg.lam_grid)
        ch, w = sel["choice"], sel["w"]
        L_d = ch["L_d"]
        tau = F.gate_threshold(A, w, L_d, cfg.gate_fpr)
        lo = max(cfg.lr_range[0], L_d + 1)
        ranking = F.rank_lr(A, r, lo, cfg.lr_range[1])
        L_star = ranking[0]["L"]
        out = cfg.wdir("fit", cfg.tag, "x").parent
        # Probe variants for the ablations (same layer, own threshold at the same benign FPR).
        w_en = F.fit_probe(A, L_d, ch["beta"], ch["lam"], ["en"])
        w_b0 = F.fit_probe(A, L_d, 0.0, ch["lam"], A.langs)
        np.savez(out / "fit.npz", w=w, w_en=w_en, w_beta0=w_b0,
                 **{f"r.{L}": r[L] for L in r})
        info = {"L_d": L_d, "beta": ch["beta"], "lam": ch["lam"], "tau": tau,
                "tau_en": F.gate_threshold(A, w_en, L_d, cfg.gate_fpr),
                "tau_beta0": F.gate_threshold(A, w_b0, L_d, cfg.gate_fpr),
                "L_star": L_star, "lr_ranking": ranking, "m_star": m_star, "m0": m0,
                "probe_selection": ch, "probe_table": sel["table"]}
        (out / "fit.json").write_text(json.dumps(info, indent=2))
        an = F.analysis(A, r, m_star, m0, w, L_d, tau, ch["beta"], ch["lam"])
        (out / "analysis.json").write_text(json.dumps(an, indent=2))
        spec = BridgeSpec(L_d=L_d, L_r=[L_star], tau=tau, c=1.0, model_id=cfg.model_id,
                          m_star={str(L_star): m_star[L_star]})
        save_bridge(out / "bridge_init", spec, w, r)
        push(cfg, out, f"fit/{cfg.tag}")
        rec["metrics"] = {"L_d": L_d, "beta": ch["beta"], "lam": ch["lam"], "tau": tau, "L_star": L_star,
                          "probe_val_auroc_min": ch["min"], "probe_val_auroc_mean": ch["mean"],
                          "D_test": an["D_test"], "D_lolo": an["D_test_leave_one_language_out"],
                          "R_test_at_L_star": an["R_test_by_layer"][L_star],
                          "margin_rel_en_at_L_star": an["margin_rel_en_by_layer"][L_star],
                          "gate_tpr_test": an["gate_tpr_test"], "gate_fpr_test": an["gate_fpr_test"],
                          "gate_fpr_xstest": an["gate_fpr_xstest"]}


def _fit(cfg: Cfg):
    d = Path(cfg.work, "fit", cfg.tag)
    if not (d / "fit.json").exists():
        d = pull(cfg, f"fit/{cfg.tag}")
    info = json.loads((d / "fit.json").read_text())
    z = np.load(d / "fit.npz")
    r = {int(k.split(".")[1]): z[k] for k in z.files if k.startswith("r.")}
    info["m_star"] = {int(k): v for k, v in info["m_star"].items()}
    return info, z, r


# ----------------------------------------------------------------------------- eval = generate + judge
def select_rows(cfg: Cfg, split: str, langs=None, labels=("harmful", "benign", "xstest_safe"), n=None):
    rows = [dict(r) for r in load_bench(cfg) if r["split"] == split and r["label"] in labels
            and (langs is None or r["lang"] in langs)]
    if n:
        keep: dict[tuple, int] = {}
        out = []
        for r in rows:
            k = (r["lang"], r["label"])
            keep[k] = keep.get(k, 0) + 1
            if keep[k] <= n:
                out.append(r)
        rows = out
    return rows


def evaluate(cfg: Cfg, S: Session, rows: list[dict], ctx=None, native: bool = False) -> list[dict]:
    from .activations import generate
    from .judge import judge_rows

    model, tok = S.lm
    outs, gates = generate(model, tok, [r["text"] for r in rows], cfg.max_new_tokens, cfg.gen_batch,
                           ctx=ctx, return_gate=True)
    for r, o, g in zip(rows, outs, gates):
        r["response"], r["gate"] = o, g
    return judge_rows(cfg, rows, S.tr, S.judge, native=native)


def do_eval(cfg: Cfg, S: Session, cond: str, ctx=None, split="test", langs=None, labels=None, n=None,
            native=False, extra: dict | None = None) -> dict:
    """Generate + judge one condition; writes eval/{tag}/{cond}_{split}.jsonl and returns its summary."""
    from .metrics import summarize

    name = f"eval_{cfg.tag}_{cond}_{split}"
    out = cfg.wdir("eval", cfg.tag, f"{cond}_{split}.jsonl")
    if not stage(cfg, name):
        if not out.exists():
            out = pull(cfg, f"eval/{cfg.tag}/{cond}_{split}.jsonl")
        return summarize(read_jsonl(out))
    with run(cfg, name) as rec:
        rows = select_rows(cfg, split, langs, labels or ("harmful", "benign", "xstest_safe"), n)
        t0 = time.time()
        rows = evaluate(cfg, S, rows, ctx=ctx, native=native)
        for r in rows:
            r["cond"] = cond
        write_jsonl(out, rows)
        push(cfg, out, f"eval/{cfg.tag}/{cond}_{split}.jsonl")
        s = summarize(rows)
        rec["metrics"] = {"n": len(rows), "seconds": round(time.time() - t0, 1), **(extra or {}),
                          **{k: v for k, v in s.items() if k != "per_lang"},
                          "asr": {l: round(d["asr"], 3) for l, d in s["per_lang"].items()},
                          "orr_benign": {l: round(d["orr_benign"], 3) for l, d in s["per_lang"].items()}}
        return s


# ----------------------------------------------------------------------------- tune
def do_tune(cfg: Cfg, S: Session):
    """Pick (L_r set, c) on validation: lowest macro ASR subject to benign over-refusal <= base + 4 pp."""
    name = f"tune_{cfg.tag}"
    out = cfg.wdir("fit", cfg.tag, "bridge", "x").parent
    if not stage(cfg, name):
        return
    with run(cfg, name) as rec:
        info, z, r = _fit(cfg)
        model, _ = S.lm
        labels = ("harmful", "benign")
        base = do_eval(cfg, S, "base", None, "val", TUNE_LANGS, labels)
        L, nL = info["L_star"], len(r)
        lr_sets = [[L], [x for x in (L - 2, L, L + 2) if info["L_d"] < x < nL]]
        table, best = [], None
        for lr in lr_sets:
            for c in cfg.c_grid:
                spec = BridgeSpec(L_d=info["L_d"], L_r=lr, tau=info["tau"], c=c, model_id=cfg.model_id,
                                  m_star={str(x): info["m_star"][x] for x in lr})
                cond = f"tune_L{'-'.join(map(str, lr))}_c{c}"
                s = do_eval(cfg, S, cond, Bridge(model, spec, z["w"], r), "val", TUNE_LANGS, labels)
                row = {"L_r": lr, "c": c, "asr": s["asr_macro"], "orr_benign": s["orr_benign_macro"],
                       "asr_per_lang": {l: d["asr"] for l, d in s["per_lang"].items()}}
                table.append(row)
                ok = row["orr_benign"] <= base["orr_benign_macro"] + 0.04
                key = (ok, -row["asr"], -c, -len(lr))
                if best is None or key > best[0]:
                    best = (key, spec, row)
        spec = best[1]
        spec.meta = {"selected_on": "val", "tune_langs": TUNE_LANGS, "base_val_asr": base["asr_macro"],
                     "base_val_orr_benign": base["orr_benign_macro"], "val_row": best[2],
                     "probe": {k: info[k] for k in ("beta", "lam")}, "gate_fpr_target": cfg.gate_fpr}
        save_bridge(out, spec, z["w"], r)
        (out / "tune_table.json").write_text(json.dumps({"base": {"asr": base["asr_macro"],
                                                                  "orr_benign": base["orr_benign_macro"]},
                                                         "grid": table}, indent=2))
        push(cfg, out, f"fit/{cfg.tag}/bridge")
        rec["metrics"] = {"chosen": best[2], "base_val_asr": base["asr_macro"],
                          "base_val_orr_benign": base["orr_benign_macro"], "grid": table}


def final_bridge(cfg: Cfg, model):
    d = Path(cfg.work, "fit", cfg.tag, "bridge")
    if not (d / "config.json").exists():
        d = pull(cfg, f"fit/{cfg.tag}/bridge")
    spec, w, r = load_bridge(d)
    return spec, w, r


def variants(cfg: Cfg, spec: BridgeSpec, w, r) -> dict:
    """Ablation bridges derived from the tuned spec."""
    info, z, r_all = _fit(cfg)
    rng = np.random.default_rng(cfg.seed)
    r_rand = {L: F.unit(rng.standard_normal(r[L].shape)) for L in spec.L_r}
    return {
        "abl_add": (replace(spec, mode="add"), w, r),
        "abl_nogate": (replace(spec, gate="always"), w, r),
        "abl_randdir": (spec, w, r_rand),
        "abl_enprobe": (replace(spec, tau=info["tau_en"]), z["w_en"], r),
        "abl_beta0": (replace(spec, tau=info["tau_beta0"]), z["w_beta0"], r),
    }


# ----------------------------------------------------------------------------- report
def do_report(cfg: Cfg):
    from .metrics import judge_agreement, paired, summarize

    with run(cfg, f"report_{cfg.tag}", mark_done=False) as rec:
        d = pull(cfg, f"eval/{cfg.tag}")
        conds = {f.stem[: -len("_test")]: read_jsonl(f) for f in sorted(d.glob("*_test.jsonl"))}
        res: dict = {"model_id": cfg.model_id, "conditions": {}, "paired_vs_base": {}}
        for c, rows in conds.items():
            res["conditions"][c] = summarize(rows)
            if c != "base" and "base" in conds:
                res["paired_vs_base"][c] = {
                    "asr": paired(conds["base"], rows, "harmful_response", "harmful"),
                    "orr_benign": paired(conds["base"], rows, "refusal", "benign"),
                    "orr_xstest": paired(conds["base"], rows, "refusal", "xstest_safe")}
        if "base" in conds:
            res["judge_validity_native_vs_pivot"] = judge_agreement(conds["base"])
        try:
            an = json.loads((pull(cfg, f"fit/{cfg.tag}") / "analysis.json").read_text())
            info = json.loads((pull(cfg, f"fit/{cfg.tag}") / "fit.json").read_text())
            if "base" in res["conditions"]:
                pl = res["conditions"]["base"]["per_lang"]
                langs = [l for l in an["langs"] if l in pl]
                asr = [pl[l]["asr"] for l in langs]
                Ls = str(info["L_star"])
                res["hypothesis"] = {
                    "spearman_asr_vs_routing_margin": F.spearman(asr, [an["margin_rel_en_by_layer"][Ls][l] for l in langs]),
                    "spearman_asr_vs_routing_auroc": F.spearman(asr, [an["R_test_by_layer"][Ls][l] for l in langs]),
                    "spearman_asr_vs_detection_auroc": F.spearman(asr, [an["D_test"][l] for l in langs]),
                    "D_test": an["D_test"], "D_lolo": an["D_test_leave_one_language_out"],
                    "R_test_at_L_star": an["R_test_by_layer"][Ls],
                    "margin_rel_en_at_L_star": an["margin_rel_en_by_layer"][Ls], "n_langs": len(langs)}
        except Exception as e:  # noqa: BLE001
            res["hypothesis_error"] = f"{type(e).__name__}: {e}"
        out = cfg.wdir("results", cfg.tag, "metrics.json")
        out.write_text(json.dumps(res, indent=2))
        lines = ["cond,lang,tier,asr,asr_lo,asr_hi,rr,orr_benign,orr_xstest,gate_harmful,gate_benign,n_harmful"]
        for c, s in res["conditions"].items():
            for l, v in s["per_lang"].items():
                lines.append(",".join(str(x) for x in (
                    c, l, v["tier"], round(v["asr"], 4), round(v["asr_ci"][0], 4), round(v["asr_ci"][1], 4),
                    round(v["rr"], 4), round(v["orr_benign"], 4), round(v["orr_xstest"], 4),
                    v["gate_harmful"], v["gate_benign"], v["asr_n"])))
        cfg.wdir("results", cfg.tag, "per_language.csv").write_text("\n".join(lines))
        push(cfg, out.parent, f"results/{cfg.tag}")
        rec["metrics"] = {c: {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v)
                              for k, v in s.items() if k != "per_lang"}
                          for c, s in res["conditions"].items()}
        rec["metrics"]["hypothesis"] = res.get("hypothesis")


# ----------------------------------------------------------------------------- orchestration
def run_a(cfg: Cfg, quick: bool = False):
    S = Session(cfg)
    do_bench(cfg, limit=24 if quick else None, S=S)
    do_acts(cfg, S)
    do_fit(cfg)
    n = 8 if quick else None
    base = do_eval(cfg, S, "base", None, "test", n=n, native=True)
    print(f"[G1] baseline ASR en={base['per_lang'].get('en', {}).get('asr')} gap_max={base.get('gap_max')} "
          f"gap_low_tier={base.get('gap_low_tier')}")
    do_tune(cfg, S)
    model, _ = S.lm
    spec, w, r = final_bridge(cfg, model)
    do_eval(cfg, S, "bridge", Bridge(model, spec, w, r), "test", n=n, extra={"spec": asdict(spec)})
    do_report(cfg)
    for name, (sp, ww, rr) in variants(cfg, spec, w, r).items():
        try:
            do_eval(cfg, S, name, Bridge(model, sp, ww, rr), "test", ABL_LANGS, ("harmful", "benign"),
                    n=n or 100)
        except Exception as e:  # noqa: BLE001 - one ablation must not sink the run
            print(f"[warn] ablation {name} failed: {type(e).__name__}: {e}")
    do_report(cfg)
    from .kits import publish_adapter

    publish_adapter(cfg)


def run_b(cfg: Cfg, quick: bool = False):
    from . import kits

    S = Session(cfg)
    n = 8 if quick else None
    for fn in (kits.safetune_baselines, kits.capability, kits.audit_redteam, kits.circuits):
        try:
            fn(cfg, S, n=n)
        except Exception as e:  # noqa: BLE001 - kit stages are independent of each other
            import traceback

            traceback.print_exc()
            print(f"[warn] {fn.__name__} failed: {type(e).__name__}: {e}")
    do_report(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bench", "acts", "fit", "tune", "report", "run_a", "run_b"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny subset, for a smoke test of the whole path")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.model:
        cfg.model_id = a.model
    if a.quick:
        import os

        assert os.environ.get("XLRB_NS"), "--quick needs XLRB_NS so it cannot mark real stages as done"
        cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_xstest_val = 12, 6, 6, 6
        cfg.ld_grid, cfg.beta_grid, cfg.lam_grid, cfg.c_grid = [8, 16], [1.0], [1e-2], [1.0]
        cfg.max_new_tokens = 48
    if a.stage == "bench":
        do_bench(cfg, a.limit, a.public)
    elif a.stage == "acts":
        do_acts(cfg, Session(cfg))
    elif a.stage == "fit":
        do_fit(cfg)
    elif a.stage == "tune":
        do_tune(cfg, Session(cfg))
    elif a.stage == "report":
        do_report(cfg)
    elif a.stage == "run_a":
        run_a(cfg, a.quick)
    elif a.stage == "run_b":
        wait_for(cfg, "bench") if not is_done(cfg, "bench") else None
        run_b(cfg, a.quick)


if __name__ == "__main__":
    main()
