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
| Test split, ten languages | Tiny Aya Global (original) | + XL-Refusal-Bridge |
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
LABELS = {"harmful": "Harmful request (from MultiJail)", "benign": "Ordinary request (from Alpaca)",
          "xstest_safe": "Safe request that sounds alarming (from XSTest)"}
ADAPTER = os.environ.get("XLRB_ADAPTER", "davidquicast/tiny-aya-xl-refusal-bridge")
ORIG, OURS = "Tiny Aya Global (original model)", "Tiny Aya Global + XL-Refusal-Bridge (our adapter)"
GATE_ON = "FIRED: the request was flagged as harmful, so the bridge was applied"
GATE_OFF = "DID NOT FIRE: the request was not flagged, so the model ran exactly as the original"
GUIDE = """
**What you are comparing.** The same model, `CohereLabs/tiny-aya-global`, answering the same prompt twice: once as
released by Cohere (left) and once with our 33 KB adapter attached (right). "Original" here means Tiny Aya Global
without the adapter; it is not the separate "Tiny Aya Base" model.

**How to read each field.**
- **Prompt / Prompt in English**: the request as sent to the model, and its English version for reference.
- **Detector**: whether our harmful-intent detector fired. If it did not, the adapter does nothing and both answers
  are identical.
- **Response / Response in English**: what the model answered, and a machine translation of it.
- **Automatic judge**: a separate model's verdict on the response. "Refused" means the model explicitly declined.
  "Harmful content" means the response gave content that serves a harmful request.
- **Outcome**: a one-line summary of what changed between the two.
"""


def _live_examples():
    want = [("zh", "harmful", "How can I buy beer"), ("it", "harmful", "can i punch people"),
            ("sw", "harmful", "I think we need to have more witch"), ("ko", "harmful", "If I have a gun")]
    out, labels = [], []
    for lang, lab, start in want:
        for e in EXAMPLES:
            if e["lang"] == lang and e["label"] == lab and e["prompt_en"].startswith(start):
                out.append([e["prompt"]]); labels.append(f"Harmful · {NAMES[lang]} · {e['prompt_en'][:48]}"); break
    for lang, lab, name in (("sw", "benign", "Ordinary"), ("jv", "benign", "Ordinary"), ("bn", "xstest_safe", "Safe but alarming"),
                            ("en", "xstest_safe", "Safe but alarming")):
        e = next(x for x in EXAMPLES if x["lang"] == lang and x["label"] == lab)
        out.append([e["prompt"]]); labels.append(f"{name} · {NAMES[lang]} · {e['prompt_en'][:48]}")
    return out, labels


try:
    import spaces

    gpu = spaces.GPU(duration=40)
except Exception:  # noqa: BLE001 - not running on a ZeroGPU Space
    gpu = lambda f: f  # noqa: E731

_state = {}


def pick(lang: str, label: str, i: int):
    rows = [e for e in EXAMPLES if e["lang"] == lang and e["label"] == label]
    e = rows[int(i) % len(rows)]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    v = lambda d: f"Refused: {yn(d['refusal'])} · Harmful content: {yn(d['harmful'])}"  # noqa: E731
    b, t = e["base_verdict"]["refusal"], e["bridge_verdict"]["refusal"]
    if not e["gate_open"]:
        out = "Detector did not fire: both responses are the same."
    elif b == t:
        out = f"Both versions {'refused' if b else 'answered'}."
    elif t:
        out = "Changed by the adapter: the original model did not refuse, with the bridge it refuses."
    else:
        out = "Changed by the adapter: the original model refused, with the bridge it does not."
    return (e["prompt"], e["prompt_en"], e["base"], e["base_en"], v(e["base_verdict"]),
            e["bridge"], e["bridge_en"], v(e["bridge_verdict"]), GATE_ON if e["gate_open"] else GATE_OFF, out)


def _load():
    if not _state:
        from huggingface_hub import snapshot_download

        from xlrb.bridge import load_bridge
        from xlrb.common import load_lm

        spec, w, r = load_bridge(snapshot_download(ADAPTER))
        model, tok = load_lm(spec.model_id, device="cpu")
        _state.update(spec=spec, w=w, r=r, model=model, tok=tok)
    return _state


try:  # load at startup, outside the GPU-timed function, so a visitor's quota is spent on generation only
    _load()
except Exception as _e:  # noqa: BLE001 - no token or no access: the recorded tab still works
    print("live model not loaded:", type(_e).__name__)


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
        return base, out[0], GATE_ON if gate[0] else GATE_OFF
    except Exception as e:  # noqa: BLE001
        msg = f"Live generation is unavailable right now ({type(e).__name__}). Use the recorded examples tab."
        return msg, msg, "n/a"


with gr.Blocks(title="XL-Refusal-Bridge") as demo:
    gr.Markdown(
        "# XL-Refusal-Bridge\n"
        "A 33 KB adapter for Cohere's `tiny-aya-global` that makes it refuse harmful requests as firmly in "
        "low-resource languages as in English, without retraining.\n\n"
        "Tiny Aya recognises harmful requests equally well in ten languages, but the signal that turns recognition "
        "into refusal is weaker in low-resource ones. A language-invariant probe gates a clamp along the refusal "
        "direction: no fine-tuning, a 33 KB adapter, and no change when the gate is closed.")
    with gr.Accordion("Results at a glance", open=True):
        gr.Markdown(RESULTS)
    with gr.Accordion("How to read this demo", open=False):
        gr.Markdown(GUIDE)
    with gr.Tab("Recorded examples"):
        gr.Markdown("Real responses saved during the evaluation (test split, sixteen per language). Choose a language "
                    "and a type of request, then move the slider. Harmful-request examples start with the cases the "
                    "adapter changed. For harmful requests we only display explicit refusals; any other response is "
                    "withheld so that this page does not publish harmful content.")
        with gr.Row():
            lang = gr.Dropdown([(NAMES[l], l) for l in LANGS], value="sw", label="1. Language of the prompt")
            label = gr.Dropdown([(v, k) for k, v in LABELS.items()], value="harmful", label="2. Type of request")
            idx = gr.Slider(0, 7, value=0, step=1, label="3. Example number")
        p = gr.Textbox(label="Prompt (as sent to the model)")
        pe = gr.Textbox(label="Prompt in English (for reference)")
        gate = gr.Textbox(label="Detector (did our harmful-intent detector fire?)")
        outc = gr.Textbox(label="Outcome")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### {ORIG}")
                b = gr.Textbox(label="Response", lines=6)
                be = gr.Textbox(label="Response in English (machine translation)", lines=6)
                bv = gr.Textbox(label="Automatic judge")
            with gr.Column():
                gr.Markdown(f"### {OURS}")
                t = gr.Textbox(label="Response", lines=6)
                te = gr.Textbox(label="Response in English (machine translation)", lines=6)
                tv = gr.Textbox(label="Automatic judge")
        outs = [p, pe, b, be, bv, t, te, tv, gate, outc]
        for c in (lang, label, idx):
            c.change(pick, [lang, label, idx], outs)
        demo.load(pick, [lang, label, idx], outs)
    with gr.Tab("Live"):
        gr.Markdown("Type a prompt, or click one of the examples below, and press **Generate**. The model answers "
                    "twice: as the original Tiny Aya Global, and with our adapter attached. The first run takes "
                    "up to a minute. Hugging Face gives visitors a small daily GPU allowance; if you see a quota "
                    "message, sign in to Hugging Face or use the recorded examples. The adapter was evaluated in English, Chinese, Italian, "
                    "Vietnamese, Arabic, Korean, Thai, Bengali, Swahili and Javanese.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=3, value="Ninawezaje kuoka mkate nyumbani?")
            n = gr.Slider(16, 256, value=96, step=8, label="Maximum length of the answer (tokens)")
        ex, exl = _live_examples()
        gr.Examples(examples=ex, example_labels=exl, inputs=[prompt], label="Examples to try (click one)")
        btn = gr.Button("Generate", variant="primary")
        g2 = gr.Textbox(label="Detector (did our harmful-intent detector fire?)", interactive=False)
        with gr.Row():
            a2 = gr.Textbox(label=ORIG, lines=10)
            b2 = gr.Textbox(label=OURS, lines=10)
        btn.click(live, [prompt, n], [a2, b2, g2])

if __name__ == "__main__":
    demo.launch()
