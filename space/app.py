"""Gradio demo: Tiny Aya with and without the gated refusal bridge.

Tab 1 browses responses recorded during the evaluation (always available).
Tab 2 runs the model live; it needs a GPU and access to the gated base model.
"""
import json
import os
from pathlib import Path

import gradio as gr

EXAMPLES = json.loads(Path(__file__).with_name("examples.json").read_text())
LANGS = sorted({e["lang"] for e in EXAMPLES}, key=["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"].index)
NAMES = {"en": "English", "zh": "Chinese", "it": "Italian", "vi": "Vietnamese", "ar": "Arabic", "ko": "Korean",
         "th": "Thai", "bn": "Bengali", "sw": "Swahili", "jv": "Javanese"}
# Lead with the cases the adapter changed: base did not refuse, bridge did.
EXAMPLES.sort(key=lambda e: not (e["label"] == "harmful" and not e["base_verdict"]["refusal"]
                                 and e["bridge_verdict"]["refusal"]))
RESULTS = """
| Test split, ten languages | Tiny Aya | + bridge |
|---|---|---|
| Harmful responses to harmful prompts | 5.4 % | **0.1 %** |
| Harmful prompts refused | 69.9 % | **95.6 %** |
| Benign prompts refused | 6.0 % | 8.2 % |
| Safe-but-alarming prompts refused (XSTest) | 17.6 % | 24.4 % |
| Worst-language gap to English | 9.3 pp | **0.7 pp** |
| Reading comprehension (Belebele) | 75.6 % | 75.6 % |

A ten-times-larger judge gives the same picture (7.5 % → 0.1 % harmful responses).
[Article and code](https://github.com/davidquicast/xl-refusal-bridge) ·
[adapter](https://huggingface.co/davidquicast/tiny-aya-xl-refusal-bridge) ·
[benchmark](https://huggingface.co/datasets/davidquicast/xl-refusal-bench)
"""
LABELS = {"harmful": "harmful (MultiJail)", "benign": "benign (Alpaca)", "xstest_safe": "safe but alarming (XSTest)"}
ADAPTER = os.environ.get("XLRB_ADAPTER", "davidquicast/tiny-aya-xl-refusal-bridge")

try:
    import spaces

    gpu = spaces.GPU(duration=120)
except Exception:  # noqa: BLE001 - not running on a ZeroGPU Space
    gpu = lambda f: f  # noqa: E731

_state = {}


def pick(lang: str, label: str, i: int):
    rows = [e for e in EXAMPLES if e["lang"] == lang and e["label"] == label]
    e = rows[int(i) % len(rows)]
    v = lambda d: f"refusal: {d['refusal']} · harmful: {d['harmful']}"  # noqa: E731
    return (e["prompt"], e["prompt_en"], e["base"], e["base_en"], v(e["base_verdict"]),
            e["bridge"], e["bridge_en"], v(e["bridge_verdict"]),
            "OPEN (clamp applied)" if e["gate_open"] else "closed (model untouched)")


def _load():
    if not _state:
        from huggingface_hub import snapshot_download

        from xlrb.bridge import load_bridge
        from xlrb.common import load_lm

        spec, w, r = load_bridge(snapshot_download(ADAPTER))
        model, tok = load_lm(spec.model_id, device="cpu")
        _state.update(spec=spec, w=w, r=r, model=model, tok=tok)
    return _state


@gpu
def live(prompt: str, max_new_tokens: int):
    try:
        import torch

        from xlrb.activations import generate
        from xlrb.bridge import Bridge

        s = _load()
        m = s["model"].to("cuda", dtype=torch.bfloat16)
        base = generate(m, s["tok"], [prompt], int(max_new_tokens), 1)[0]
        br = Bridge(m, s["spec"], s["w"], s["r"])
        out, gate = generate(m, s["tok"], [prompt], int(max_new_tokens), 1, ctx=br, return_gate=True)
        return base, out[0], "OPEN (clamp applied)" if gate[0] else "closed (model untouched)"
    except Exception as e:  # noqa: BLE001
        msg = f"Live generation is unavailable here ({type(e).__name__}). Use the recorded examples tab."
        return msg, msg, "n/a"


with gr.Blocks(title="XL-Refusal-Bridge") as demo:
    gr.Markdown(
        "# XL-Refusal-Bridge\n"
        "Tiny Aya recognises harmful requests equally well in ten languages, but the signal that turns recognition "
        "into refusal is weaker in low-resource ones. A language-invariant probe gates a clamp along the refusal "
        "direction: no fine-tuning, a 33 KB adapter, and no change when the gate is closed.")
    with gr.Accordion("Results at a glance", open=True):
        gr.Markdown(RESULTS)
    with gr.Tab("Recorded examples"):
        gr.Markdown("Responses from the evaluation run (test split), sixteen per language. Harmful-prompt examples "
                    "start with the cases the bridge changed (base did not refuse, bridge did). For harmful prompts "
                    "only explicit refusals are shown; any other response is withheld. Verdicts are the automatic "
                    "judge's. The gate line shows whether the detector fired; when it is closed the two responses "
                    "are identical by construction.")
        with gr.Row():
            lang = gr.Dropdown([(NAMES[l], l) for l in LANGS], value="sw", label="Language")
            label = gr.Dropdown([(v, k) for k, v in LABELS.items()], value="harmful", label="Prompt class")
            idx = gr.Slider(0, 7, value=0, step=1, label="Example")
        p, pe = gr.Textbox(label="Prompt"), gr.Textbox(label="Prompt (English)")
        gate = gr.Textbox(label="Gate")
        with gr.Row():
            with gr.Column():
                b, be, bv = gr.Textbox(label="Base", lines=6), gr.Textbox(label="Base (English)", lines=6), gr.Textbox(label="Judge")
            with gr.Column():
                t, te, tv = gr.Textbox(label="Bridge", lines=6), gr.Textbox(label="Bridge (English)", lines=6), gr.Textbox(label="Judge")
        outs = [p, pe, b, be, bv, t, te, tv, gate]
        for c in (lang, label, idx):
            c.change(pick, [lang, label, idx], outs)
        demo.load(pick, [lang, label, idx], outs)
    with gr.Tab("Live"):
        gr.Markdown("Runs the model with and without the bridge. Needs a GPU and access to the gated base model.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt (any of the ten evaluated languages)", lines=3,
                                value="Ninawezaje kuoka mkate nyumbani?")
            n = gr.Slider(16, 256, value=96, step=8, label="max new tokens")
        btn = gr.Button("Generate", variant="primary")
        g2 = gr.Textbox(label="Gate", interactive=False)
        with gr.Row():
            a2 = gr.Textbox(label="Tiny Aya (base)", lines=10)
            b2 = gr.Textbox(label="Tiny Aya + bridge", lines=10)
        btn.click(live, [prompt, n], [a2, b2, g2])

if __name__ == "__main__":
    demo.launch()
