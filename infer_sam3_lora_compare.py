#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import Sam3Model, Sam3Processor


def get_hf_token() -> str | None:
    token_path = Path("hf_token.txt")
    if token_path.exists():
        tok = token_path.read_text().strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
            return tok
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_mask_png(mask_bool: np.ndarray, path: Path) -> None:
    arr = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(arr).save(path)


def resize_mask_to_image(mask_bool: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize bool mask to match original image size.

    image_size is (width, height), PIL-style.
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_u8)
    mask_img = mask_img.resize(image_size, Image.NEAREST)
    return np.array(mask_img) > 0


def build_color_overlay(image_rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    image_rgb: HxWx3 uint8
    mask_bool: HxW bool, same spatial size as image_rgb
    """
    if mask_bool.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match image shape {image_rgb.shape[:2]}"
        )

    overlay = image_rgb.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[..., 0] = 255  # red overlay

    out = overlay.copy()
    out[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color[mask_bool]
    return np.clip(out, 0, 255).astype(np.uint8)


def side_by_side(images: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Image.Image:
    arrs = [Image.fromarray(x) for x in images]
    widths, heights = zip(*(im.size for im in arrs))
    canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))

    x = 0
    for im in arrs:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def logits_to_binary_mask(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    logits: [1, 1, H, W] or [1, H, W]
    returns bool HxW
    """
    if logits.ndim == 4:
        logits = logits[:, 0]
    probs = torch.sigmoid(logits)
    mask = probs[0].detach().cpu().numpy() > threshold
    return mask


@torch.no_grad()
def run_semantic_seg(model, processor, image: Image.Image, prompt: str, device: str, threshold: float) -> Dict:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    if not hasattr(outputs, "semantic_seg"):
        raise RuntimeError("Model outputs do not contain semantic_seg")

    sem = outputs.semantic_seg  # [B, 1, H, W]
    pred_mask_small = logits_to_binary_mask(sem, threshold=threshold)

    result = {
        "semantic_seg_shape": tuple(sem.shape),
        "mask_small": pred_mask_small,
    }

    if hasattr(outputs, "pred_masks"):
        result["pred_masks_shape"] = tuple(outputs.pred_masks.shape)

    return result


def load_lora_model(adapter_dir: str, token: str | None, device: str):
    base_model = Sam3Model.from_pretrained("facebook/sam3", token=token)
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir)
    lora_model = lora_model.to(device)
    lora_model.eval()
    return lora_model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--adapter-dir", default="sam3_lora_out/best", help="Path to LoRA adapter directory")
    parser.add_argument("--prompt", default="particle", help="Text prompt")
    parser.add_argument("--out-dir", default="lora_compare_out", help="Directory to save comparison outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold on semantic_seg")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    token = get_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("hf token present:", bool(token))
    print("image:", args.image)
    print("adapter_dir:", args.adapter_dir)
    print("prompt:", args.prompt)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    image_size = image.size  # (width, height)

    print("original image size:", image_size)

    print("Loading base model...")
    base_model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(device)
    base_model.eval()

    print("Loading processor...")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)

    print("Running base model...")
    base_result = run_semantic_seg(base_model, processor, image, args.prompt, device, args.threshold)

    print("Loading LoRA adapter...")
    lora_model = load_lora_model(args.adapter_dir, token, device)

    print("Running LoRA model...")
    lora_result = run_semantic_seg(lora_model, processor, image, args.prompt, device, args.threshold)

    # Resize masks to original image size for visualization
    base_mask = resize_mask_to_image(base_result["mask_small"], image_size)
    lora_mask = resize_mask_to_image(lora_result["mask_small"], image_size)

    # Save resized masks
    save_mask_png(base_mask, out_dir / "base_mask.png")
    save_mask_png(lora_mask, out_dir / "lora_mask.png")

    # Save overlays
    base_overlay = build_color_overlay(image_np, base_mask, alpha=0.45)
    lora_overlay = build_color_overlay(image_np, lora_mask, alpha=0.45)
    Image.fromarray(base_overlay).save(out_dir / "base_overlay.png")
    Image.fromarray(lora_overlay).save(out_dir / "lora_overlay.png")

    # Save side-by-side image: original | base overlay | lora overlay
    comparison = side_by_side((image_np, base_overlay, lora_overlay))
    comparison.save(out_dir / "comparison.png")

    # Save metadata
    report = {
        "image": args.image,
        "adapter_dir": args.adapter_dir,
        "prompt": args.prompt,
        "device": device,
        "original_image_size_wh": [image_size[0], image_size[1]],
        "base": {
            "semantic_seg_shape": base_result["semantic_seg_shape"],
            "pred_masks_shape": base_result.get("pred_masks_shape"),
            "positive_pixels_resized": int(base_mask.sum()),
        },
        "lora": {
            "semantic_seg_shape": lora_result["semantic_seg_shape"],
            "pred_masks_shape": lora_result.get("pred_masks_shape"),
            "positive_pixels_resized": int(lora_mask.sum()),
        },
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved outputs to:", out_dir.resolve())
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()