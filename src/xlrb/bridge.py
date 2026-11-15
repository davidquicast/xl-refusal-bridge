"""The gated refusal bridge: a closed-form, gradient-free activation intervention.

    gate   g(x) = 1[ w . h^{L_d}(x_last) > tau ]                       (language-invariant harm probe)
    clamp  h^{L} <- h^{L} + g(x) * max(0, c * m*_L - r_L . h^{L}) r_L   for L in L_r  (idempotent)

`w`, `tau` come from an invariance-regularised LDA probe; `r_L` is the unit refusal direction and
`m*_L` the mean projection of English harmful prompts at layer L. Parameters: 2d + 2 per clamp layer.
The gate is read once, at the last prompt token, and held for the whole generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class BridgeSpec:
    L_d: int                       # detection layer (hook on the output of block L_d)
    L_r: list[int]                 # clamp layers, all > L_d
    tau: float
    c: float = 1.0
    mode: str = "clamp"            # "clamp" (ours) | "add" (additive ablation: h += c * m* * r)
    gate: str = "probe"            # "probe" | "always" (ungated ablation)
    positions: str = "gen"         # "gen": last prompt token + generated tokens | "all"
    m_star: dict[str, float] = field(default_factory=dict)   # str(layer) -> scalar
    model_id: str = ""
    meta: dict = field(default_factory=dict)


class Bridge:
    """Context manager that installs the bridge hooks on a HF causal LM."""

    def __init__(self, model, spec: BridgeSpec, w: np.ndarray, r: dict[int, np.ndarray]):
        import torch

        from .common import get_layers

        assert all(L > spec.L_d for L in spec.L_r) or spec.gate == "always", "clamp layers must follow L_d"
        self.spec = spec
        self.layers = get_layers(model)
        dev = next(model.parameters()).device
        self.w = torch.tensor(np.asarray(w), dtype=torch.float32, device=dev)
        self.r = {int(L): torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) for L, v in r.items()}
        self.g = None            # (B, 1, 1) gate, set by the first forward of each generate() call
        self.scores = None       # (B,) raw probe scores of the last batch, for logging
        self._handles: list = []

    # -- hooks -------------------------------------------------------------------------------
    @staticmethod
    def _h(out):
        return out if not isinstance(out, (tuple, list)) else out[0]

    def _detect(self, _mod, _inp, out):
        if self.g is None:
            h = self._h(out)[:, -1, :].float()
            s = h @ self.w
            self.scores = s.detach().cpu()
            self.g = (s > self.spec.tau).float()[:, None, None]

    def _make_clamp(self, L: int):
        import torch

        r, m = self.r[L], float(self.spec.m_star[str(L)])
        c, mode, pos = self.spec.c, self.spec.mode, self.spec.positions

        def clamp(_mod, _inp, out):
            h = self._h(out)
            if self.spec.gate == "always":
                g = 1.0
            elif self.g is None:
                return out
            else:
                g = self.g
            if mode == "clamp":
                p = (h.float() @ r)[..., None]
                delta = torch.clamp(c * m - p, min=0.0) * r
            else:
                delta = (c * m * r).expand(h.shape[0], h.shape[1], -1)
            delta = (g * delta).to(h.dtype)
            if pos == "gen" and h.shape[1] > 1:      # prompt pass: touch only the last prompt token
                mask = torch.zeros(1, h.shape[1], 1, dtype=h.dtype, device=h.device)
                mask[:, -1] = 1
                delta = delta * mask
            new = h + delta
            return new if not isinstance(out, (tuple, list)) else (new,) + tuple(out[1:])

        return clamp

    # -- lifecycle ---------------------------------------------------------------------------
    def reset(self):
        self.g, self.scores = None, None

    def __enter__(self):
        self.reset()
        if self.spec.gate == "probe":
            self._handles.append(self.layers[self.spec.L_d].register_forward_hook(self._detect))
        for L in self.spec.L_r:
            self._handles.append(self.layers[L].register_forward_hook(self._make_clamp(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ----------------------------------------------------------------------------- persistence
def save_bridge(path: str | Path, spec: BridgeSpec, w: np.ndarray, r: dict[int, np.ndarray]) -> None:
    from safetensors.numpy import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {"w": np.asarray(w, dtype=np.float32)}
    for L in spec.L_r:
        tensors[f"r.{L}"] = np.asarray(r[L], dtype=np.float32)
    save_file(tensors, str(path / "bridge.safetensors"))
    (path / "config.json").write_text(json.dumps(asdict(spec), indent=2))


def load_bridge(path: str | Path):
    from safetensors.numpy import load_file

    path = Path(path)
    spec = BridgeSpec(**json.loads((path / "config.json").read_text()))
    t = load_file(str(path / "bridge.safetensors"))
    return spec, t["w"], {int(k.split(".")[1]): v for k, v in t.items() if k.startswith("r.")}
