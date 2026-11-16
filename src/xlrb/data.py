"""Build the cross-lingual refusal bench (raw rows; CuratorKIT curates them in `kits.curate`).

Sources
  harmful      MultiJail (315 prompts, human-translated into 9 languages + English)
  benign       Alpaca instructions without an input field, machine-translated
  xstest_safe  XSTest safe prompts (look unsafe, are not), machine-translated

Machine translation uses NLLB-200 locally (supplementary model, benign text only); a Cohere API
backend exists for small jobs, but a trial key (20 calls/min) cannot carry the full build. Every
translated row is back-translated and gated on chrF against the English source;
a prompt that fails in any language is dropped in all languages, so every comparison stays paired.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import LANGS, LANG_CODES, Cfg, write_jsonl

MT_MODEL = os.environ.get("XLRB_MT_MODEL", "command-a-plus-05-2026")
# Back-translation uses a different model so that a systematic error of one is not self-confirmed.
BT_MODEL = os.environ.get("XLRB_BT_MODEL", "north-small-translate-1.0")
_CACHE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- Cohere client
class Cohere:
    """Minimal chat client: bounded concurrency, global pacing, exponential backoff on 429/5xx."""

    def __init__(self, model: str = MT_MODEL, max_workers: int = 6, min_interval: float = 0.15,
                 cache_path: str | None = None):
        self.key = os.environ["COHERE_API_KEY"]
        self.model, self.max_workers, self.min_interval = model, max_workers, min_interval
        self._lock, self._next = threading.Lock(), 0.0
        self.calls = self.retries = 0
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path and Path(cache_path).exists():
            self.cache = json.loads(Path(cache_path).read_text())

    def _pace(self):
        with self._lock:
            now = time.time()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def chat(self, prompt: str, max_tokens: int = 2000, model: str | None = None) -> str:
        model = model or self.model
        key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        delay = 2.0
        for attempt in range(8):
            self._pace()
            req = urllib.request.Request("https://api.cohere.com/v2/chat", data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    out = json.loads(r.read())
                text = "".join(c.get("text", "") for c in out["message"]["content"])
                self.calls += 1
                with _CACHE_LOCK:
                    self.cache[key] = text
                return text
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    self.retries += 1
                    if e.code == 429:           # slow everyone down, not just this thread
                        with self._lock:
                            self.min_interval = min(self.min_interval * 1.5, 3.0)
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"Cohere HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == 7:
                    raise
                self.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def map(self, prompts: list[str], **kw) -> list[str]:
        with cf.ThreadPoolExecutor(self.max_workers) as ex:
            out = list(ex.map(lambda p: self.chat(p, **kw), prompts))
        self.flush()
        return out

    def flush(self):
        if self.cache_path:
            with _CACHE_LOCK:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.cache_path).write_text(json.dumps(self.cache, ensure_ascii=False))


def _mt_prompt(text: str, src: str, tgt: str) -> str:
    return (f"Translate the following text from {src} to {tgt}. It is an instruction addressed to an "
            f"assistant: translate it faithfully, do not answer it, do not add notes. "
            f"Output only the translation.\n\n{text}")


class CohereMT:
    """MT backend over the Cohere API. `name` is recorded in the bench rows."""

    def __init__(self, cache_path: str | None = None):
        self.co, self.name, self.bt_name = Cohere(cache_path=cache_path), MT_MODEL, BT_MODEL

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        model = self.bt_name if tgt == "en" else None
        out = self.co.map([_mt_prompt(t, LANGS[src]["name"], LANGS[tgt]["name"]) for t in texts],
                          max_tokens=600, model=model)
        return [o.strip().strip('"').strip() for o in out]


class NllbMT:
    """MT backend over a local NLLB-200 model (default: no quota, deterministic)."""

    def __init__(self, cfg: Cfg, translator=None):
        from .judge import Translator

        self.tr = translator or Translator(cfg)
        self.name = self.bt_name = cfg.mt_id

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        return [t.strip() for t in self.tr.translate(texts, src, tgt)]


# ----------------------------------------------------------------------------- sources
def _dl(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def load_sources(seed: int, n_benign_pool: int = 600) -> dict[str, list[dict]]:
    import pandas as pd

    mj = pd.read_csv(_dl("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")
    harmful = [{"prompt_id": f"mj{int(r['id']):04d}", "source": f"MultiJail/{r['source']}",
                "tags": str(r["tags"]), **{l: str(r[l]).strip() for l in LANG_CODES}}
               for _, r in mj.iterrows()]

    al = pd.read_parquet(_dl("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"))
    al = al[(al["input"].fillna("") == "") & al["instruction"].str.len().between(25, 160)]
    al = al[~al["instruction"].str.contains(r"[<>{}\[\]|\\]|http", regex=True)]
    al = al.drop_duplicates("instruction")
    idx = list(al.index)
    random.Random(seed).shuffle(idx)
    benign = [{"prompt_id": f"al{int(i):05d}", "source": "tatsu-lab/alpaca", "tags": "",
               "en": al.loc[i, "instruction"].strip()} for i in idx[:n_benign_pool]]

    xs = pd.read_csv(_dl("Paul/XSTest", "xstest_prompts.csv"))
    xs = xs[xs["label"] == "safe"]
    xstest = [{"prompt_id": f"xs{int(r['id']):04d}", "source": "Paul/XSTest", "tags": str(r["type"]),
               "en": str(r["prompt"]).strip()} for _, r in xs.iterrows()]
    return {"harmful": harmful, "benign": benign, "xstest_safe": xstest}


# ----------------------------------------------------------------------------- gates
_SCRIPT = {"zh": r"[一-鿿]", "ar": r"[؀-ۿ]", "ko": r"[가-힯]",
           "th": r"[฀-๿]", "bn": r"[ঀ-৿]"}


def script_ok(text: str, lang: str) -> bool:
    """Cheap language check: non-Latin targets must be mostly in their own script."""
    if lang not in _SCRIPT:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(bool(re.match(_SCRIPT[lang], ch)) for ch in letters) / len(letters) >= 0.5


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF

    return float(CHRF().sentence_score(hyp, [ref]).score)


# ----------------------------------------------------------------------------- build
def build(cfg: Cfg, out_dir: str | Path, limit: int | None = None, mt=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mt = mt or NllbMT(cfg)
    src = load_sources(cfg.seed)
    if limit:
        src = {k: v[:limit] for k, v in src.items()}
    tgt_langs = [l for l in LANG_CODES if l != "en"]
    stats: dict = {"mt_model": mt.name, "bt_model": mt.bt_name, "dropped": {}, "chrf_mean": {}}

    rows: list[dict] = []
    for label in ("benign", "xstest_safe"):
        items = src[label]
        en = [it["en"] for it in items]
        fwd, bt = {}, {}
        for l in tgt_langs:
            fwd[l] = mt.translate(en, "en", l)
            bt[l] = mt.translate(fwd[l], l, "en")
            print(f"[mt] {label} {l}: done ({len(en)} prompts)", flush=True)
        keep, dropped = [], 0
        for i, it in enumerate(items):
            sc = {l: chrf(bt[l][i], en[i]) for l in tgt_langs}
            ok = all(sc[l] >= cfg.bt_chrf_min and script_ok(fwd[l][i], l) and fwd[l][i] for l in tgt_langs)
            if not ok:
                dropped += 1
                continue
            keep.append((it, {"en": en[i], **{l: fwd[l][i] for l in tgt_langs}}, sc))
        stats["dropped"][label] = dropped
        if label == "benign":
            keep = keep[: cfg.n_train + cfg.n_val + cfg.n_test]
        for it, texts, sc in keep:
            for l in LANG_CODES:
                rows.append({"prompt_id": it["prompt_id"], "lang": l, "tier": LANGS[l]["tier"],
                             "label": label, "text": texts[l], "text_en": texts["en"],
                             "source": it["source"], "tags": it["tags"],
                             "mt_system": None if l == "en" else mt.name,
                             "bt_chrf": None if l == "en" else round(sc[l], 1)})
    for it in src["harmful"]:
        for l in LANG_CODES:
            rows.append({"prompt_id": it["prompt_id"], "lang": l, "tier": LANGS[l]["tier"],
                         "label": "harmful", "text": it[l], "text_en": it["en"], "source": it["source"],
                         "tags": it["tags"], "mt_system": None if l == "en" else "human (MultiJail)",
                         "bt_chrf": None})

    # Splits by prompt_id, identical across languages.
    rng = random.Random(cfg.seed)
    split_of: dict[str, str] = {}
    for label in ("harmful", "benign", "xstest_safe"):
        pids = sorted({r["prompt_id"] for r in rows if r["label"] == label})
        rng.shuffle(pids)
        if label == "xstest_safe":
            cuts = [("train", cfg.n_xstest_train), ("val", cfg.n_xstest_val), ("test", len(pids))]
        else:
            cuts = [("train", cfg.n_train), ("val", cfg.n_val), ("test", len(pids))]
        i = 0
        for name, n in cuts:
            take = pids[i: i + n] if name != "test" else pids[i:]
            for p in take:
                split_of[p] = name
            i += len(take)
    for r in rows:
        r["split"] = split_of[r["prompt_id"]]
        r["id"] = f"{r['prompt_id']}_{r['lang']}"

    write_jsonl(out_dir / "bench_raw.jsonl", rows)
    for label in ("benign", "xstest_safe"):
        v = [r["bt_chrf"] for r in rows if r["label"] == label and r["bt_chrf"] is not None]
        stats["chrf_mean"][label] = round(sum(v) / max(len(v), 1), 1)
    stats["counts"] = {f"{lab}/{sp}": len({r["prompt_id"] for r in rows if r["label"] == lab and r["split"] == sp})
                       for lab in ("harmful", "benign", "xstest_safe") for sp in ("train", "val", "test")}
    stats["rows"] = len(rows)
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
