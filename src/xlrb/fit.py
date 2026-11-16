"""Closed-form fitting and the H1/H2 analysis. NumPy only: no model, no gradients.

Inputs are last-prompt-token residuals cached by `activations.py`:
    acts[(lang, label)] = (ids: list[str], X: float32 [N, n_layers + 1, d])
where X[:, i + 1] is the output of decoder block i (X[:, 0] is the embedding output).
"""
from __future__ import annotations

import numpy as np

from .common import LANG_CODES, LANGS


# ----------------------------------------------------------------------------- primitives
def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney U)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def lda_probe(XH: list[np.ndarray], XB: list[np.ndarray], beta: float, lam: float) -> np.ndarray:
    """Invariance-regularised LDA.

        w = (S_W + beta * S_lang + lam * tr(S_W)/d * I)^-1 (mu_H - mu_B)

    XH / XB: one [n_l, d] array per training language. S_W is the pooled within-(language, class)
    covariance; S_lang is the covariance of the language centroids, which down-weights directions
    that encode language identity so the probe transfers to unseen languages.
    """
    d = XH[0].shape[1]
    SW = np.zeros((d, d))
    n = 0
    cents = []
    for a, b in zip(XH, XB):
        for X in (a, b):
            Xc = X - X.mean(0, keepdims=True)
            SW += Xc.T @ Xc
            n += len(X)
        cents.append((a.mean(0) + b.mean(0)) / 2)
    SW /= max(n - 2 * len(XH), 1)
    C = np.stack(cents)
    Cc = C - C.mean(0, keepdims=True)
    SL = Cc.T @ Cc / max(len(cents) - 1, 1) if len(cents) > 1 else np.zeros((d, d))
    A = SW + beta * SL + lam * (np.trace(SW) / d) * np.eye(d)
    muH = np.mean([a.mean(0) for a in XH], axis=0)
    muB = np.mean([b.mean(0) for b in XB], axis=0)
    return unit(np.linalg.solve(A, muH - muB))


# ----------------------------------------------------------------------------- data access
class Acts:
    def __init__(self, acts: dict, split_of: dict[str, str]):
        self.acts, self.split_of = acts, split_of

    def get(self, lang: str, label: str, split: str, layer: int) -> np.ndarray:
        """Residual at the output of block `layer` for one (lang, label, split) cell."""
        ids, X = self.acts[(lang, label)]
        idx = [i for i, k in enumerate(ids) if self.split_of[k] == split]
        return np.asarray(X[idx, layer + 1], dtype=np.float64)

    @property
    def n_layers(self) -> int:
        return next(iter(self.acts.values()))[1].shape[1] - 1

    @property
    def langs(self) -> list[str]:
        return [l for l in LANG_CODES if (l, "harmful") in self.acts]


# ----------------------------------------------------------------------------- fitting
def refusal_dirs(A: Acts, langs: list[str] = ("en",)) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
    """Per layer: unit diff-of-means direction, m* = mean harmful projection, m0 = mean benign projection."""
    r, m_star, m0 = {}, {}, {}
    for L in range(A.n_layers):
        H = np.concatenate([A.get(l, "harmful", "train", L) for l in langs])
        B = np.concatenate([A.get(l, "benign", "train", L) for l in langs])
        r[L] = unit(H.mean(0) - B.mean(0))
        m_star[L] = float((H @ r[L]).mean())
        m0[L] = float((B @ r[L]).mean())
    return r, m_star, m0


def fit_probe(A: Acts, L: int, beta: float, lam: float, langs: list[str], hard: bool = True) -> np.ndarray:
    """`hard=True` adds XSTest-safe training prompts (look unsafe, are safe) to the negative class,
    so the probe has to separate harmful intent from merely alarming vocabulary."""
    neg = []
    for l in langs:
        B = A.get(l, "benign", "train", L)
        if hard and (l, "xstest_safe") in A.acts:
            B = np.concatenate([B, A.get(l, "xstest_safe", "train", L)])
        neg.append(B)
    return lda_probe([A.get(l, "harmful", "train", L) for l in langs], neg, beta, lam)


def per_lang_auroc(A: Acts, v: np.ndarray, L: int, split: str, neg: str = "benign") -> dict[str, float]:
    """neg: "benign", "xstest_safe", or "pool" (both)."""
    labs = ("benign", "xstest_safe") if neg == "pool" else (neg,)
    return {l: auroc(A.get(l, "harmful", split, L) @ v,
                     np.concatenate([A.get(l, lab, split, L) @ v for lab in labs if (l, lab) in A.acts]))
            for l in A.langs}


def select_probe(A: Acts, ld_grid, beta_grid, lam_grid, train_langs: list[str] | None = None) -> dict:
    """Pick (L_d, beta, lam) maximising the worst-language validation AUROC (ties: mean)."""
    train_langs = train_langs or A.langs
    best, table = None, []
    for L in ld_grid:
        if L >= A.n_layers:
            continue
        for beta in beta_grid:
            for lam in lam_grid:
                w = fit_probe(A, L, beta, lam, train_langs)
                au = per_lang_auroc(A, w, L, "val", neg="pool")
                row = {"L_d": L, "beta": beta, "lam": lam, "min": min(au.values()),
                       "mean": float(np.mean(list(au.values())))}
                table.append(row)
                key = (row["min"], row["mean"])
                if best is None or key > best[0]:
                    best = (key, row, w)
    return {"choice": best[1], "w": best[2], "table": table}


def gate_threshold(A: Acts, w: np.ndarray, L: int, fpr: float,
                   pool: tuple[str, ...] = ("benign", "xstest_safe")) -> float:
    s = np.concatenate([A.get(l, lab, "val", L) @ w for l in A.langs for lab in pool if (l, lab) in A.acts])
    return float(np.quantile(s, 1 - fpr))


def rank_lr(A: Acts, r: dict[int, np.ndarray], lo: int, hi: int) -> list[dict]:
    """Rank candidate clamp layers by how well the English refusal direction separates English val."""
    rows = []
    # The last hidden state can be post-final-norm (not a block output), so it is never a candidate.
    for L in range(lo, min(hi, A.n_layers - 2) + 1):
        rows.append({"L": L, "auroc_en": auroc(A.get("en", "harmful", "val", L) @ r[L],
                                                 A.get("en", "benign", "val", L) @ r[L])})
    return sorted(rows, key=lambda x: -x["auroc_en"])


# ----------------------------------------------------------------------------- analysis (H1 / H2)
def analysis(A: Acts, r_en, m_star, m0, w, L_d: int, tau: float, beta: float, lam: float) -> dict:
    out: dict = {"langs": A.langs, "tiers": {l: LANGS[l]["tier"] for l in A.langs}}
    nL = A.n_layers
    # H1: detection AUROC of the pooled probe at L_d (test), and with each language held out.
    out["D_test"] = per_lang_auroc(A, w, L_d, "test")
    out["D_test_xstest"] = per_lang_auroc(A, w, L_d, "test", neg="xstest_safe")
    lolo = {}
    for l in A.langs:
        others = [x for x in A.langs if x != l]
        wl = fit_probe(A, L_d, beta, lam, others)
        lolo[l] = auroc(A.get(l, "harmful", "test", L_d) @ wl, A.get(l, "benign", "test", L_d) @ wl)
    out["D_test_leave_one_language_out"] = lolo
    w_en = fit_probe(A, L_d, beta, lam, ["en"])
    out["D_test_english_only_probe"] = per_lang_auroc(A, w_en, L_d, "test")
    out["D_test_xstest_english_only_probe"] = per_lang_auroc(A, w_en, L_d, "test", neg="xstest_safe")
    w_nh = fit_probe(A, L_d, beta, lam, A.langs, hard=False)
    out["D_test_xstest_no_hard_negatives"] = per_lang_auroc(A, w_nh, L_d, "test", neg="xstest_safe")
    # Gate behaviour at the chosen threshold (pure activation-level; no generation needed).
    out["gate_tpr_test"] = {l: float((A.get(l, "harmful", "test", L_d) @ w > tau).mean()) for l in A.langs}
    out["gate_fpr_test"] = {l: float((A.get(l, "benign", "test", L_d) @ w > tau).mean()) for l in A.langs}
    out["gate_fpr_xstest"] = {l: float((A.get(l, "xstest_safe", "test", L_d) @ w > tau).mean()) for l in A.langs}
    # H2: routing. Per layer and language: AUROC along r_en, and the margin relative to English.
    R, M, COS = {}, {}, {}
    for L in range(nL):
        rl = r_en[L]
        R[L] = per_lang_auroc(A, rl, L, "test")
        den = m_star[L] - m0[L] + 1e-12
        M[L] = {l: float(((A.get(l, "harmful", "test", L) @ rl).mean()
                          - (A.get(l, "benign", "test", L) @ rl).mean()) / den) for l in A.langs}
        COS[L] = {l: float(unit(A.get(l, "harmful", "train", L).mean(0)
                                - A.get(l, "benign", "train", L).mean(0)) @ rl) for l in A.langs}
    out["R_test_by_layer"], out["margin_rel_en_by_layer"], out["cos_r_lang_r_en_by_layer"] = R, M, COS
    # Detection by layer with a per-layer probe (same beta, lam) for the D-vs-R figure.
    out["D_test_by_layer"] = {L: per_lang_auroc(A, fit_probe(A, L, beta, lam, A.langs), L, "test")
                              for L in range(0, nL, 2)}
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))
