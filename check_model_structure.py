"""
Script to check Qwen2.5-VL model structure for cross-attention layers.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import config
import utils

print("="*80)
print("Loading Qwen2.5-VL Model to Check Structure")
print("="*80)

# Load model
print(f"\nLoading model: {config.MODEL_NAME}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=config.MODEL_TORCH_DTYPE,
    device_map="cpu",  # Load on CPU for inspection
    attn_implementation="eager"
)

print("Model loaded successfully!")

# Print detailed structure
utils.print_model_structure(model)

print("\nDone!")
