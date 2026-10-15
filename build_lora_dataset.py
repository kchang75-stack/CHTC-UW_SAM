#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr > 0


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    out = (mask.astype(np.uint8) * 255)
    Image.fromarray(out).save(path)


def build_union_mask(image_dir: Path) -> np.ndarray:
    inst_json = image_dir / "instances.json"
    with open(inst_json, "r") as f:
        meta = json.load(f)

    instances = meta["instances"]
    if not instances:
        raise ValueError(f"No instances in {inst_json}")

    first_mask = load_binary_mask(image_dir / instances[0]["mask_file"])
    union = np.zeros_like(first_mask, dtype=bool)

    for inst in instances:
        mask = load_binary_mask(image_dir / inst["mask_file"])
        union |= mask

    return union


def collect_image_dirs(clean_root: Path) -> Iterable[Path]:
    for p in sorted(clean_root.iterdir()):
        if p.is_dir() and (p / "instances.json").exists():
            yield p


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True, help="aggregated_clean directory")
    parser.add_argument("--out-root", default="lora_dataset", help="output dataset root")
    parser.add_argument("--val-every", type=int, default=5, help="every Nth image goes to val")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    out_root = Path(args.out_root)

    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    image_dirs = list(collect_image_dirs(clean_root))
    if not image_dirs:
        raise RuntimeError(f"No image directories found in {clean_root}")

    for idx, image_dir in enumerate(image_dirs):
        split = "val" if (idx % args.val_every == 0) else "train"

        stem = image_dir.name
        original_candidates = list(image_dir.glob("*_original.png"))
        if len(original_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly 1 original image in {image_dir}, found {len(original_candidates)}"
            )

        original_path = original_candidates[0]
        union_mask = build_union_mask(image_dir)

        out_image = out_root / split / "images" / f"{stem}.png"
        out_mask = out_root / split / "masks" / f"{stem}.png"

        shutil.copy2(original_path, out_image)
        save_binary_mask(union_mask, out_mask)

    print(f"Wrote dataset to {out_root}")


if __name__ == "__main__":
    main()