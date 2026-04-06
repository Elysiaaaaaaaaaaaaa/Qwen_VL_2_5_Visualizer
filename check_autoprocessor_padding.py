import argparse
from pprint import pprint
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor


def _collect_attrs(obj: Any, names) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for n in names:
        if hasattr(obj, n):
            try:
                out[n] = getattr(obj, n)
            except Exception as e:
                out[n] = f"<error reading attr: {e}>"
    return out


def _try_get_config_dict(obj: Any) -> Optional[Dict[str, Any]]:
    for attr in ("to_dict", "get_config_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                val = fn()
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
    cfg = getattr(obj, "config", None)
    if cfg is not None:
        to_dict = getattr(cfg, "to_dict", None)
        if callable(to_dict):
            try:
                val = to_dict()
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Qwen2.5-VL AutoProcessor/image_processor for padding behavior."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model id or local path for AutoProcessor.from_pretrained(...)",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to a local image file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt used to build chat template input",
    )
    parser.add_argument(
        "--content_order",
        type=str,
        default="Image → Text",
        choices=["Image → Text", "Text → Image"],
        help="Match your app's content order",
    )
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Pass padding=True into processor(...). (Note: this always pads text; image padding depends on image_processor.)",
    )
    args = parser.parse_args()

    print("== Loading AutoProcessor ==")
    processor = AutoProcessor.from_pretrained(args.model)
    print("processor class:", type(processor))

    print("\n== Processor config (if available) ==")
    cfg = _try_get_config_dict(processor)
    if cfg is None:
        print("<no dict-like config found>")
    else:
        keys = sorted(k for k in cfg.keys() if any(s in k.lower() for s in ("pad", "resize", "size", "crop")))
        if keys:
            print("filtered keys (pad/resize/size/crop):")
            for k in keys:
                print(f"  {k}: {cfg.get(k)}")
        else:
            print("<no pad/resize/size/crop keys found in processor config dict>")

    if not hasattr(processor, "image_processor"):
        print("\n== image_processor ==")
        print("<processor has no image_processor attribute; cannot inspect image padding directly>")
        return

    ip = processor.image_processor
    print("\n== image_processor ==")
    print("image_processor class:", type(ip))

    print("\n== image_processor common attrs (pad/resize/size/crop related) ==")
    common_attr_names = [
        "do_pad",
        "do_resize",
        "do_center_crop",
        "do_normalize",
        "pad_size",
        "pad_to_multiple_of",
        "padding_value",
        "padding_mode",
        "size",
        "crop_size",
        "resample",
        "image_mean",
        "image_std",
    ]
    pprint(_collect_attrs(ip, common_attr_names), sort_dicts=True)

    print("\n== image_processor config dict (filtered) ==")
    ip_cfg = _try_get_config_dict(ip)
    if ip_cfg is None:
        print("<no dict-like config found>")
    else:
        keys = sorted(k for k in ip_cfg.keys() if any(s in k.lower() for s in ("pad", "resize", "size", "crop")))
        if keys:
            for k in keys:
                print(f"  {k}: {ip_cfg.get(k)}")
        else:
            print("<no pad/resize/size/crop keys found in image_processor config dict>")

    print("\n== Run one preprocessing pass to inspect outputs ==")
    image = Image.open(args.image).convert("RGB")
    print("original image (W,H):", image.size)

    if args.content_order == "Image → Text":
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}, {"type": "image", "image": image}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prefer qwen_vl_utils (same as your app) if installed/available.
    image_inputs = [image]
    video_inputs = None
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as e:
        print("note: qwen_vl_utils.process_vision_info not available/failed, fallback to raw PIL image.")
        print("  error:", e)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=bool(args.padding),
        return_tensors="pt",
    )

    print("processor output keys:", sorted(list(inputs.keys())))

    pv = inputs.get("pixel_values", None)
    grid = inputs.get("image_grid_thw", None)
    attn_mask = inputs.get("attention_mask", None)

    if attn_mask is not None:
        print("attention_mask shape:", tuple(attn_mask.shape), " (text padding shows up here)")
        if attn_mask.numel() > 0:
            pad_ratio = (attn_mask == 0).float().mean().item()
            print(f"attention_mask padding ratio (zeros): {pad_ratio:.4f}")

    if pv is not None:
        print("pixel_values shape:", tuple(pv.shape), " (B,C,H,W)")
        if pv.dim() == 4:
            _, _, h, w = pv.shape
            print("pixel_values derived (W,H):", (int(w), int(h)))

    if grid is not None and isinstance(grid, torch.Tensor) and grid.numel() >= 3:
        t, h, w = grid[0].tolist()
        print("image_grid_thw[0] (T,H,W):", (t, h, w))
        # Qwen2.5-VL uses 14px patches; 2x2 spatial merge => 28px per merged token,
        # but grid_thw is typically in patch units (14px). This gives a useful sanity-check.
        proc_h_14 = int(h * 14)
        proc_w_14 = int(w * 14)
        print("grid-derived processed (W,H) assuming 14px patches:", (proc_w_14, proc_h_14))
        print("multiple of 28:", (proc_w_14 % 28 == 0, proc_h_14 % 28 == 0))

    print("\nDone.")


if __name__ == "__main__":
    main()

