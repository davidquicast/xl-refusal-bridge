"""Figures and markdown tables for the write-up, from the files the report stage produces."""
from __future__ import annotations

import json
from pathlib import Path

from .common import LANG_CODES, LANGS

TIER_COLOR = {"ref": "#444444", "high": "#2a7fbf", "mid": "#d98c1f", "low": "#c0392b"}
COND_NAME = {"base": "Tiny Aya (base)", "bridge": "+ gated bridge (ours)", "st_refdir": "SafeTune refusal direction",
             "st_cast": "SafeTune CAST", "abl_add": "ablation: additive", "abl_nogate": "ablation: no gate",
             "abl_randdir": "ablation: random direction", "abl_enprobe": "ablation: English-only probe",
             "abl_beta0": "ablation: beta = 0", "abl_nohardneg": "ablation: no hard negatives in the probe"}


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def tables(res: dict) -> str:
    conds = res["conditions"]
    out = ["### Aggregate (test split)\n",
           "| Condition | ASR macro ↓ | Refusal on harmful ↑ | Over-refusal benign ↓ | Over-refusal XSTest ↓ | Gap max (pp) ↓ | Gap low-tier (pp) ↓ | Languages |",
           "|---|---|---|---|---|---|---|---|"]
    order = [c for c in COND_NAME if c in conds] + [c for c in conds if c not in COND_NAME]
    for c in order:
        s = conds[c]
        out.append(f"| {COND_NAME.get(c, c)} | {pct(s['asr_macro'])} | {pct(s['rr_macro'])} | {pct(s['orr_benign_macro'])} | "
                   f"{pct(s['orr_xstest_macro'])} | {pct(s.get('gap_max'))} | {pct(s.get('gap_low_tier'))} | {len(s['per_lang'])} |")
    if "base" in conds and "bridge" in conds:
        b, t = conds["base"]["per_lang"], conds["bridge"]["per_lang"]
        pv = res.get("paired_vs_base", {}).get("bridge", {}).get("asr", {})
        out += ["\n### Per language: base → bridge (test split)\n",
                "| Lang | Tier | ASR % (95% CI) base | ASR % (95% CI) bridge | Holm p (McNemar) | Refusal % | Over-refusal benign % | Over-refusal XSTest % | Gate on harmful % |",
                "|---|---|---|---|---|---|---|---|---|"]
        for l in LANG_CODES:
            if l not in b or l not in t:
                continue
            ci = lambda d: f"{pct(d['asr'])} ({pct(d['asr_ci'][0])}–{pct(d['asr_ci'][1])})"  # noqa: E731
            p = pv.get(l, {}).get("p_holm")
            out.append(f"| {l} | {b[l]['tier']} | {ci(b[l])} | {ci(t[l])} | {'n/a' if p is None else f'{p:.3g}'} | "
                       f"{pct(b[l]['rr'])} → {pct(t[l]['rr'])} | {pct(b[l]['orr_benign'])} → {pct(t[l]['orr_benign'])} | "
                       f"{pct(b[l]['orr_xstest'])} → {pct(t[l]['orr_xstest'])} | {pct(t[l].get('gate_harmful'))} |")
    return "\n".join(out)


def make(res: dict, analysis: dict | None, tune: dict | None, out_dir: str | Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    conds = res["conditions"]
    if "base" in conds:
        langs = [l for l in LANG_CODES if l in conds["base"]["per_lang"]]
        show = [c for c in ("base", "st_refdir", "st_cast", "bridge") if c in conds]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (key, title) in zip(axes, (("asr", "Attack success rate on harmful prompts (↓)"),
                                           ("orr_benign", "Over-refusal on benign prompts (↓)"),
                                           ("orr_xstest", "Over-refusal on XSTest-safe prompts (↓)"))):
            wd = 0.8 / len(show)
            for i, c in enumerate(show):
                pl = conds[c]["per_lang"]
                xs = [j + i * wd for j, l in enumerate(langs) if l in pl]
                ys = [100 * pl[l][key] for l in langs if l in pl]
                err = None
                if key == "asr":
                    err = [[100 * (pl[l]["asr"] - pl[l]["asr_ci"][0]) for l in langs if l in pl],
                           [100 * (pl[l]["asr_ci"][1] - pl[l]["asr"]) for l in langs if l in pl]]
                ax.bar(xs, ys, wd, yerr=err, capsize=2, label=COND_NAME.get(c, c))
            ax.set_xticks([j + 0.4 - wd / 2 for j in range(len(langs))])
            ax.set_xticklabels([f"{l}\n({LANGS[l]['tier']})" for l in langs], fontsize=8)
            ax.set_ylabel("%")
            ax.set_title(title, fontsize=10)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "before_after.png", dpi=160)
        plt.close(fig)
        made.append("before_after.png")
    if analysis:
        langs = analysis["langs"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for name, key, ax in (("Detection: probe AUROC by layer", "D_test_by_layer", axes[0]),
                              ("Routing: AUROC along the English refusal direction", "R_test_by_layer", axes[1]),
                              ("Routing margin relative to English", "margin_rel_en_by_layer", axes[2])):
            d = analysis[key]
            Ls = sorted(int(k) for k in d)
            for l in langs:
                ax.plot(Ls, [d[str(L)][l] for L in Ls], color=TIER_COLOR[LANGS[l]["tier"]], alpha=0.8, lw=1.3,
                        label=LANGS[l]["tier"])
            ax.set_xlabel("layer")
            ax.set_title(name, fontsize=10)
        h, lab = axes[0].get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        axes[0].legend(uniq.values(), uniq.keys(), fontsize=8, title="resource tier")
        fig.tight_layout()
        fig.savefig(out_dir / "detect_vs_route.png", dpi=160)
        plt.close(fig)
        made.append("detect_vs_route.png")
    if tune:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        for row in tune["grid"]:
            ax.scatter(100 * row["orr_benign"], 100 * row["asr"], s=40)
            ax.annotate(f"L{row['L_r']} c={row['c']}", (100 * row["orr_benign"], 100 * row["asr"]), fontsize=7)
        ax.scatter(100 * tune["base"]["orr_benign"], 100 * tune["base"]["asr"], marker="*", s=160, color="k", label="base")
        ax.set_xlabel("over-refusal on benign, val (%)")
        ax.set_ylabel("ASR, val (%)")
        ax.set_title("Validation trade-off (en, bn, sw, jv)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "tune_tradeoff.png", dpi=160)
        plt.close(fig)
        made.append("tune_tradeoff.png")
    return made


def main(results_dir: str, fit_dir: str, out_dir: str):
    res = json.loads(Path(results_dir, "metrics.json").read_text())
    an = Path(fit_dir, "analysis.json")
    tn = Path(fit_dir, "bridge", "tune_table.json")
    made = make(res, json.loads(an.read_text()) if an.exists() else None,
                json.loads(tn.read_text()) if tn.exists() else None, out_dir)
    Path(out_dir, "tables.md").write_text(tables(res))
    print("figures:", made)


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:4])
