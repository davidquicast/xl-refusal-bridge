"""Shared constants, config, Hub message bus, run logging and model loading."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# code -> (name, NLLB code, Belebele file stem, MultiJail resource tier)
LANGS: dict[str, dict[str, str]] = {
    "en": {"name": "English", "nllb": "eng_Latn", "belebele": "eng_Latn", "tier": "ref"},
    "zh": {"name": "Chinese", "nllb": "zho_Hans", "belebele": "zho_Hans", "tier": "high"},
    "it": {"name": "Italian", "nllb": "ita_Latn", "belebele": "ita_Latn", "tier": "high"},
    "vi": {"name": "Vietnamese", "nllb": "vie_Latn", "belebele": "vie_Latn", "tier": "high"},
    "ar": {"name": "Arabic", "nllb": "arb_Arab", "belebele": "arb_Arab", "tier": "mid"},
    "ko": {"name": "Korean", "nllb": "kor_Hang", "belebele": "kor_Hang", "tier": "mid"},
    "th": {"name": "Thai", "nllb": "tha_Thai", "belebele": "tha_Thai", "tier": "mid"},
    "bn": {"name": "Bengali", "nllb": "ben_Beng", "belebele": "ben_Beng", "tier": "low"},
    "sw": {"name": "Swahili", "nllb": "swh_Latn", "belebele": "swh_Latn", "tier": "low"},
    "jv": {"name": "Javanese", "nllb": "jav_Latn", "belebele": "jav_Latn", "tier": "low"},
}
LANG_CODES = list(LANGS)
# Languages that are never used for fitting; they only appear in the out-of-distribution test.
EXTRA_NLLB = {"de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "ja": "jpn_Jpan",
              "ru": "rus_Cyrl"}


def nllb_code(lang: str) -> str:
    return LANGS[lang]["nllb"] if lang in LANGS else EXTRA_NLLB[lang]
LABELS = ("harmful", "benign", "xstest_safe")
# Languages the PolyGuard judge was trained on (used for the native-language cross-check).
POLYGUARD_LANGS = {"en", "zh", "it", "ar", "ko", "th"}


@dataclass
class Cfg:
    hf_user: str = os.environ.get("XLRB_HF_USER", "")
    model_id: str = os.environ.get("XLRB_MODEL", "CohereLabs/tiny-aya-global")
    judge_id: str = os.environ.get("XLRB_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
    judge2_id: str = "ToxicityPrompts/PolyGuard-Qwen-Smol"   # second opinion on refusal only
    mt_id: str = "facebook/nllb-200-distilled-1.3B"
    seed: int = 0
    rev: str = "r2"              # protocol revision: r2 = hard negatives in the probe + stricter refusal rubric
    # earlier revision whose generations may be re-judged instead of regenerated
    reuse_tag: str = os.environ.get("XLRB_REUSE_TAG", "")   # default: the same model, without revision
    n_train: int = 115
    n_val: int = 50
    n_test: int = 150
    n_xstest_train: int = 80     # XSTest-safe prompts used as hard negatives for the probe
    n_xstest_val: int = 40
    bt_chrf_min: float = 25.0
    max_new_tokens: int = 128
    gen_batch: int = 32
    act_batch: int = 32
    judge_batch: int = 16
    mt_batch: int = 48
    gate_fpr: float = 0.05       # on the pooled benign + XSTest-safe validation prompts
    ld_grid: list[int] = field(default_factory=lambda: list(range(6, 25, 2)))
    lr_range: tuple[int, int] = (14, 30)
    beta_grid: list[float] = field(default_factory=lambda: [0.0, 1.0, 10.0])
    lam_grid: list[float] = field(default_factory=lambda: [1e-3, 1e-2, 1e-1])
    c_grid: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25, 1.5])
    work: str = os.environ.get("XLRB_WORK", "/content/xlrb_work")

    # Hub repos. bench/adapter/logs are public deliverables, ops is the private bus.
    @property
    def ops_repo(self) -> str:
        return f"{self.hf_user}/xlrb-ops"

    @property
    def bench_repo(self) -> str:
        return f"{self.hf_user}/xl-refusal-bench"

    @property
    def adapter_repo(self) -> str:
        return f"{self.hf_user}/tiny-aya-xl-refusal-bridge"

    @property
    def logs_repo(self) -> str:
        return f"{self.hf_user}/xl-refusal-bridge-logs"

    @property
    def tag(self) -> str:
        """Namespace for per-model artifacts, so replications never collide."""
        return self.model_id.split("/")[-1] + ("-" + self.rev if self.rev else "")

    def wdir(self, *parts: str) -> Path:
        p = Path(self.work, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_cfg(path: str | None = None) -> Cfg:
    cfg = Cfg()
    if path and Path(path).exists():
        import yaml

        for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if not cfg.hf_user:
        from huggingface_hub import whoami

        cfg.hf_user = whoami()["name"]
    return cfg


# ----------------------------------------------------------------------------- Hub bus
# XLRB_BUS_DIR switches the bus to a local directory (offline dry runs and tests).
BUS_DIR = os.environ.get("XLRB_BUS_DIR")
# XLRB_NS prefixes every path on the ops bus, so a smoke test cannot mark real stages as done.
NS = os.environ.get("XLRB_NS", "")


def _ns(path_in_repo: str, repo: str | None) -> str:
    return f"ns/{NS}/{path_in_repo}" if NS and repo is None else path_in_repo


def _bus_path(cfg: "Cfg", repo: str | None, path_in_repo: str) -> Path:
    return Path(BUS_DIR, (repo or cfg.ops_repo).replace("/", "__"), path_in_repo)


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def ensure_repo(repo_id: str, repo_type: str = "dataset", private: bool = True) -> None:
    if BUS_DIR:
        return
    _api().create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)


def push(cfg: Cfg, local: str | Path, path_in_repo: str, repo: str | None = None,
         repo_type: str = "dataset", msg: str | None = None) -> None:
    local = Path(local)
    path_in_repo = _ns(path_in_repo, repo)
    if BUS_DIR:
        import shutil

        dst = _bus_path(cfg, repo, path_in_repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local.is_dir():
            shutil.copytree(local, dst, dirs_exist_ok=True)
        else:
            shutil.copyfile(local, dst)
        return
    repo = repo or cfg.ops_repo
    msg = msg or f"chore: upload {path_in_repo}"
    for attempt in range(4):
        try:
            if local.is_dir():
                _api().upload_folder(folder_path=str(local), path_in_repo=path_in_repo, repo_id=repo,
                                     repo_type=repo_type, commit_message=msg)
            else:
                _api().upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=repo,
                                   repo_type=repo_type, commit_message=msg)
            return
        except Exception as e:  # noqa: BLE001 - network flakiness on Colab
            if attempt == 3:
                raise
            print(f"[hub] upload retry {attempt + 1}: {type(e).__name__}: {e}")
            time.sleep(10 * (attempt + 1))


def pull(cfg: Cfg, path_in_repo: str, repo: str | None = None, repo_type: str = "dataset") -> Path:
    """Download a file or a folder prefix from the bus into the work dir; returns the local path."""
    path_in_repo = _ns(path_in_repo, repo)
    if BUS_DIR:
        return _bus_path(cfg, repo, path_in_repo)
    from huggingface_hub import snapshot_download

    repo = repo or cfg.ops_repo
    root = Path(cfg.work, "_hub", repo.replace("/", "__"))
    snapshot_download(repo, repo_type=repo_type, local_dir=str(root),
                      allow_patterns=[path_in_repo, f"{path_in_repo}/*"])
    return root / path_in_repo


def exists(cfg: Cfg, path_in_repo: str, repo: str | None = None, repo_type: str = "dataset") -> bool:
    path_in_repo = _ns(path_in_repo, repo)
    if BUS_DIR:
        return _bus_path(cfg, repo, path_in_repo).exists()
    try:
        return _api().file_exists(repo or cfg.ops_repo, path_in_repo, repo_type=repo_type)
    except Exception:  # noqa: BLE001
        return False


def done_key(stage: str) -> str:
    return f"done/{stage}.json"


def is_done(cfg: Cfg, stage: str) -> bool:
    return exists(cfg, done_key(stage))


def wait_for(cfg: Cfg, stage: str, poll: int = 60, timeout: int = 6 * 3600) -> None:
    t0 = time.time()
    while not is_done(cfg, stage):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"stage {stage} never completed")
        print(f"[bus] waiting for stage '{stage}' ({int(time.time() - t0)}s)")
        time.sleep(poll)


# ----------------------------------------------------------------------------- run logging
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("XLRB_CODE_SHA", "unknown")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for m in ("torch", "transformers", "numpy", "safetune", "circuitkit", "auditkit", "curatorkit"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    return out


def _hardware() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
    except Exception:  # noqa: BLE001
        pass
    return {"gpu": None}


@contextlib.contextmanager
def run(cfg: Cfg, stage: str, parents: list[str] | None = None, mark_done: bool = True):
    """Context manager that records one stage as a run.json and a DONE marker on the bus.

    Schema: run_id, stage, git_sha, utc_start, utc_end, model_id, config_hash, seed, hardware,
    lib_versions, metrics, status, parent_run_ids, error.
    """
    rec: dict[str, Any] = {
        "run_id": f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{stage}_{cfg.hash()[:7]}",
        "stage": stage, "git_sha": _git_sha(), "utc_start": time.strftime("%FT%TZ", time.gmtime()),
        "model_id": cfg.model_id, "config_hash": cfg.hash(), "seed": cfg.seed,
        "hardware": _hardware(), "lib_versions": _versions(), "metrics": {}, "status": "running",
        "parent_run_ids": parents or [],
    }
    try:
        yield rec
        rec["status"] = "ok"
    except BaseException as e:  # noqa: BLE001 - record then re-raise
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec["utc_end"] = time.strftime("%FT%TZ", time.gmtime())
        p = cfg.wdir("runs", rec["run_id"], "run.json")
        p.write_text(json.dumps(rec, indent=2, default=str))
        folder = "runs" if rec["status"] == "ok" else "failed"
        try:
            push(cfg, p, f"{folder}/{rec['run_id']}/run.json")
            if rec["status"] == "ok" and mark_done:
                push(cfg, p, done_key(stage))
        except Exception as e:  # noqa: BLE001
            print(f"[bus] could not upload run record: {e}")
        print(f"[run] {stage}: {rec['status']} metrics={json.dumps(rec['metrics'], default=str)[:600]}")


# ----------------------------------------------------------------------------- jsonl
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- model
def load_lm(model_id: str, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_layers(model) -> list:
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers"), ("layers",)):
        m = model
        try:
            for a in path:
                m = getattr(m, a)
            return list(m)
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


def encode(tok, prompts: list[str], device="cpu"):
    """One user turn through the chat template (same tokens for extraction and generation)."""
    if getattr(tok, "chat_template", None):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    else:
        enc = tok(prompts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


_BENCH: dict = {}


def load_bench(cfg: Cfg) -> list[dict]:
    """Bench rows from the local work dir, else from the ops bus."""
    key = (cfg.ops_repo, NS)
    if key not in _BENCH:        # the bus copy is authoritative (a local copy can be a stale split)
        _BENCH[key] = read_jsonl(pull(cfg, "bench/bench.jsonl"))
    return _BENCH[key]
