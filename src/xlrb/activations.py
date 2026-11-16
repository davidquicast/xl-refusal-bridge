"""Cache last-prompt-token residuals for every layer, and run plain or hooked generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import LABELS, LANG_CODES, Cfg, encode


def cache_acts(cfg: Cfg, model, tok, rows: list[dict], out_dir: str | Path, batch: int | None = None) -> dict:
    """Writes {lang}_{label}.npy (float16 [N, n_layers + 1, d]) and {lang}_{label}.ids.txt."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = batch or cfg.act_batch
    dev = next(model.parameters()).device
    n = 0
    for lang in LANG_CODES:
        for label in LABELS:
            cell = [r for r in rows if r["lang"] == lang and r["label"] == label]
            f = out_dir / f"{lang}_{label}.npy"
            if not cell or f.exists():
                continue
            order = sorted(range(len(cell)), key=lambda i: len(cell[i]["text"]))
            chunks: dict[int, np.ndarray] = {}
            for s in range(0, len(order), batch):
                idx = order[s: s + batch]
                enc = encode(tok, [cell[i]["text"] for i in idx], dev)
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states
                last = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()  # [B, L+1, d]
                for j, i in enumerate(idx):
                    chunks[i] = last[j]
            X = np.stack([chunks[i] for i in range(len(cell))]).astype(np.float16)
            np.save(f, X)
            (out_dir / f"{lang}_{label}.ids.txt").write_text("\n".join(r["id"] for r in cell))
            n += len(cell)
            print(f"[acts] {lang}/{label}: {X.shape}")
    return {"prompts": n}


def selfcheck(model, tok, layer: int) -> float:
    """Max abs difference between a forward hook on block `layer` and hidden_states[layer + 1].
    The whole method assumes these are the same tensor; a large value means the indexing is off."""
    import torch

    from .common import get_layers

    dev = next(model.parameters()).device
    enc = encode(tok, ["How do I bake bread?", "Name three primary colors, please."], dev)
    cap = {}
    h = get_layers(model)[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("o", (o if not isinstance(o, (tuple, list)) else o[0])[:, -1].float()))
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    h.remove()
    return float((cap["o"] - hs[layer + 1][:, -1].float()).abs().max())


def load_acts(acts_dir: str | Path) -> dict:
    acts_dir = Path(acts_dir)
    out = {}
    for f in sorted(acts_dir.glob("*.npy")):
        lang, label = f.stem.split("_", 1)
        ids = (acts_dir / f"{f.stem}.ids.txt").read_text().split("\n")
        out[(lang, label)] = (ids, np.load(f, mmap_mode="r"))
    return out


def generate(model, tok, prompts: list[str], max_new_tokens: int = 128, batch: int = 32, ctx=None,
             return_gate: bool = False):
    """Greedy batched generation. `ctx` is an optional context manager (e.g. a Bridge, or a SafeTune
    wrapped model) that is entered around each batch; if it has `.reset()` it is reset per batch."""
    import torch

    dev = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs: list[str | None] = [None] * len(prompts)
    gates: list[float | None] = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s: s + batch]
        enc = encode(tok, [prompts[i] for i in idx], dev)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                  pad_token_id=tok.pad_token_id)
        with torch.no_grad():
            if ctx is None:
                gen = model.generate(**enc, **kw)
            else:
                with ctx:
                    if hasattr(ctx, "reset"):
                        ctx.reset()
                    gen = model.generate(**enc, **kw)
                    g = getattr(ctx, "g", None)
                    if g is not None:
                        for j, i in enumerate(idx):
                            gates[i] = float(g[j].item())
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            outs[i] = texts[j].strip()
    return (outs, gates) if return_gate else outs


def first_token_kl(model, tok, prompts: list[str], ctx, batch: int = 16) -> list[float]:
    """KL(base || steered) of the first generated token, per prompt. Computed batch by batch on the
    device: the full [N, vocab] matrices do not fit in host memory for a 250k-token vocabulary."""
    import torch

    dev = next(model.parameters()).device
    out: list[float] = []
    for s in range(0, len(prompts), batch):
        enc = encode(tok, prompts[s: s + batch], dev)
        with torch.no_grad():
            P = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
            with ctx:
                if hasattr(ctx, "reset"):
                    ctx.reset()
                Q = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
        out += (P.exp() * (P - Q)).sum(-1).cpu().tolist()
    return out


def first_token_logprobs(model, tok, prompts: list[str], batch: int = 32, ctx=None):
    """Log-probabilities of the first generated token, [N, V] float32 on CPU (for the KL metric)."""
    import torch

    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(prompts), batch):
        enc = encode(tok, prompts[s: s + batch], dev)
        with torch.no_grad():
            if ctx is None:
                lg = model(**enc).logits[:, -1]
            else:
                with ctx:
                    if hasattr(ctx, "reset"):
                        ctx.reset()
                    lg = model(**enc).logits[:, -1]
        out.append(torch.log_softmax(lg.float(), -1).cpu())
    return torch.cat(out)
