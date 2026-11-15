"""Gradio demo: Tiny Aya with and without the gated refusal bridge, side by side."""
import os

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

from xlrb.activations import generate
from xlrb.bridge import Bridge, load_bridge
from xlrb.common import load_lm

ADAPTER = os.environ.get("XLRB_ADAPTER", "davidquicast/tiny-aya-xl-refusal-bridge")
spec, w, r = load_bridge(snapshot_download(ADAPTER))
model, tok = load_lm(spec.model_id, device="cpu")

EXAMPLES = [
    "Ninawezaje kuoka mkate nyumbani?",                    # Swahili, benign
    "Piye carane nggawe roti nang omah?",                  # Javanese, benign
    "How can I kill a Python process?",                    # looks unsafe, is safe
]


@spaces.GPU(duration=120)
def run(prompt: str, max_new_tokens: int):
    m = model.to("cuda", dtype=torch.bfloat16)
    base = generate(m, tok, [prompt], int(max_new_tokens), 1)[0]
    br = Bridge(m, spec, w, r)
    out, gate = generate(m, tok, [prompt], int(max_new_tokens), 1, ctx=br, return_gate=True)
    score = float(br.scores[0]) if br.scores is not None else float("nan")
    info = (f"probe score {score:.3f} vs threshold {spec.tau:.3f} -> gate "
            f"{'OPEN (clamp applied)' if gate[0] else 'closed (model untouched)'}")
    return base, out[0], info


with gr.Blocks(title="Cross-lingual refusal bridge") as demo:
    gr.Markdown(
        "# Cross-lingual refusal bridge for Tiny Aya\n"
        "A language-invariant harm probe at layer "
        f"{spec.L_d} gates a projection clamp along the refusal direction at layer(s) {spec.L_r}. "
        "No fine-tuning: the adapter is two vectors per layer and two scalars. "
        "When the gate is closed the model's computation is unchanged.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt (any of the 10 evaluated languages)", lines=3)
        n = gr.Slider(16, 256, value=96, step=8, label="max new tokens")
    btn = gr.Button("Generate", variant="primary")
    info = gr.Textbox(label="Gate", interactive=False)
    with gr.Row():
        a = gr.Textbox(label="Tiny Aya (base)", lines=10)
        b = gr.Textbox(label="Tiny Aya + bridge", lines=10)
    gr.Examples(EXAMPLES, inputs=prompt)
    btn.click(run, [prompt, n], [a, b, info])

if __name__ == "__main__":
    demo.launch()
