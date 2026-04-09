# !pip -q install -U transformers accelerate qwen-vl-utils torchvision

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

SEED = 42
NUM_SAMPLES = 20
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
SAVE_DIR = "/kaggle/working/bos_top5_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)

if device == "cpu":
    model = model.to(device)

model.eval()

# fixed here
text_cfg = model.config.text_config
hidden_size = text_cfg.hidden_size
num_layers = text_cfg.num_hidden_layers

print("hidden_size:", hidden_size)
print("num_layers:", num_layers)

dataset = CIFAR10(
    root="/kaggle/working/cifar10_data",
    train=False,
    download=True
)

indices = list(range(len(dataset)))
random.shuffle(indices)
selected_indices = indices[:NUM_SAMPLES]

layer_dim_scores = torch.zeros(num_layers, hidden_size, dtype=torch.float64)

for sample_id, idx in enumerate(selected_indices):
    image, label = dataset[idx]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please describe this image briefly."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states
    bos_pos = 0

    # hidden_states[0] is embedding output
    # hidden_states[1] ... hidden_states[num_layers] are transformer layers
    for layer_idx in range(num_layers):
        h = hidden_states[layer_idx + 1][0, bos_pos, :].detach().float().cpu()
        rms = torch.sqrt(torch.mean(h ** 2) + 1e-12)
        layer_dim_scores[layer_idx] += (torch.abs(h) / rms).double()

    print(f"[{sample_id+1:02d}/{NUM_SAMPLES}] done")

summary_lines = []

for layer_idx in range(num_layers):
    scores = layer_dim_scores[layer_idx].numpy()
    top5_idx = np.argsort(scores)[-5:][::-1]
    top5_scores = scores[top5_idx]

    line = f"Layer {layer_idx+1:02d}: " + ", ".join(
        [f"dim {int(d)} -> {float(s):.4f}" for d, s in zip(top5_idx, top5_scores)]
    )
    summary_lines.append(line)

    plt.figure(figsize=(16, 8))
    bars = plt.bar([str(int(d)) for d in top5_idx], top5_scores)

    plt.title(
        f"Layer {layer_idx+1} | BOS Top-5 Dimensions by Cumulative Normalized Activation",
        fontsize=16
    )
    plt.xlabel("Dimension Index")
    plt.ylabel("Cumulative Normalized Activation")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for b, s in zip(bars, top5_scores):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"layer_{layer_idx+1:02d}_bos_top5.png"), dpi=200)
    plt.close()

with open(os.path.join(SAVE_DIR, "top5_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nDone. Results saved to:", SAVE_DIR)