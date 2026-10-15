#!/usr/bin/env python3
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from region_features import compute_region_features_for_all


def load_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    return arr > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """
    intersection / min(area_a, area_b)
    """
    inter = np.logical_and(a, b).sum()
    a_area = int(a.sum())
    b_area = int(b.sum())
    denom = min(a_area, b_area)
    if denom == 0:
        return 0.0
    return float(inter) / float(denom)


def bbox_iou(box1, box2) -> float:
    if box1 is None or box2 is None:
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def source_priority(source: str) -> int:
    # higher is better
    order = {
        "split_child_kept": 3,
        "kept_parent_after_failed_child_filter": 2,
        "kept_single": 1,
    }
    return order.get(source or "", 0)


def score_instance(inst: Dict, feats) -> Tuple:
    """
    Bigger tuple = better mask.
    Ordered by:
      1) preferred source
      2) circularity
      3) solidity
      4) lower aspect ratio penalty
      5) smaller area as final tie-breaker
    """
    return (
        source_priority(inst.get("source")),
        float(feats.circularity),
        float(feats.solidity),
        -float(feats.aspect_ratio),
        -float(inst.get("area_px", 0)),
    )


def masks_to_label_image(bin_masks: List[np.ndarray]) -> np.ndarray:
    """
    Build uint16 label image where 0=background and 1..N are instance ids.
    Larger masks first so smaller masks stay visible.
    """
    if not bin_masks:
        raise ValueError("No masks provided to masks_to_label_image")

    h, w = bin_masks[0].shape
    label = np.zeros((h, w), dtype=np.uint32)

    order = sorted(range(len(bin_masks)), key=lambda i: int(bin_masks[i].sum()), reverse=True)
    for new_id, i in enumerate(order, start=1):
        label[bin_masks[i]] = new_id

    return label.astype(np.uint16)


def save_label_png(label_u16: np.ndarray, path: Path) -> None:
    Image.fromarray(label_u16, mode="I;16").save(path)


def save_overlay_from_label(
    image_rgb: np.ndarray,
    label_u16: np.ndarray,
    out_path: Path,
    alpha: float = 0.43,
    seed: int = 123,
) -> None:
    label = label_u16.astype(np.int32)
    max_id = int(label.max())

    if max_id == 0:
        Image.fromarray(image_rgb).save(out_path)
        return

    rng = np.random.default_rng(seed)
    color_map = np.zeros((max_id + 1, 3), dtype=np.uint8)
    color_map[1:] = rng.integers(0, 256, size=(max_id, 3), dtype=np.uint8)

    color_mask = color_map[label]
    overlay = (
        image_rgb.astype(np.float32) * (1.0 - alpha)
        + color_mask.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)


def rle_encode_binary_mask(mask: np.ndarray) -> Dict:
    h, w = mask.shape
    pixels = mask.T.flatten().astype(np.uint8)
    counts: List[int] = []
    prev = 0
    run = 0

    for p in pixels:
        p = int(p)
        if p == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = p

    counts.append(run)
    return {"size": [h, w], "counts": counts}


def safe_bbox_from_mask(mask: np.ndarray) -> Tuple[Optional[List[int]], int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = int(xs.size)
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1], area


def dedup_one_image(
    image_root: Path,
    out_root: Path,
    iou_thresh: float = 0.75,
    contain_thresh: float = 0.90,
    bbox_gate: float = 0.10,
    overlay_alpha: float = 0.43,
) -> Dict:
    instances_json = image_root / "instances.json"
    if not instances_json.exists():
        raise FileNotFoundError(f"Missing {instances_json}")

    with open(instances_json, "r") as f:
        meta = json.load(f)

    image_name = Path(meta["image_file"]).stem
    original_candidates = meta["instances"]

    rgb_path = image_root / f"{image_name}_original.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing original image file: {rgb_path}")
    rgb = np.array(Image.open(rgb_path).convert("RGB"))

    # Load masks and instance records
    masks: List[np.ndarray] = []
    items: List[Dict] = []
    for inst in original_candidates:
        mask_path = image_root / inst["mask_file"]
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        mb = load_mask(mask_path)
        masks.append(mb)
        items.append(dict(inst))

    # Compute features on current final masks
    feats_list = compute_region_features_for_all(
        rgb,
        [{"segmentation": m} for m in masks],
    )
    feats_by_idx = {f.idx: f for f in feats_list}

    n = len(items)
    drop = set()
    decisions = []

    for i in range(n):
        if i in drop or i not in feats_by_idx:
            continue

        for j in range(i + 1, n):
            if j in drop or j not in feats_by_idx:
                continue

            box_iou = bbox_iou(items[i].get("bbox_xywh"), items[j].get("bbox_xywh"))
            if box_iou < bbox_gate:
                continue

            iou = mask_iou(masks[i], masks[j])
            cont = containment(masks[i], masks[j])

            if iou >= iou_thresh or cont >= contain_thresh:
                fi = feats_by_idx[i]
                fj = feats_by_idx[j]

                si = score_instance(items[i], fi)
                sj = score_instance(items[j], fj)

                if si >= sj:
                    keep_idx, drop_idx = i, j
                else:
                    keep_idx, drop_idx = j, i

                if drop_idx not in drop:
                    drop.add(drop_idx)
                    decisions.append(
                        {
                            "keep_id": items[keep_idx]["id"],
                            "drop_id": items[drop_idx]["id"],
                            "keep_source": items[keep_idx].get("source"),
                            "drop_source": items[drop_idx].get("source"),
                            "iou": round(iou, 4),
                            "containment": round(cont, 4),
                            "bbox_iou": round(box_iou, 4),
                        }
                    )

    kept_items: List[Dict] = []
    kept_masks: List[np.ndarray] = []
    for idx, inst in enumerate(items):
        if idx not in drop:
            kept_items.append(inst)
            kept_masks.append(masks[idx])

    # Prepare output dirs
    image_out = out_root / image_root.name
    inst_out = image_out / "instances"
    inst_out.mkdir(parents=True, exist_ok=True)

    # Renumber and resave masks
    new_instances: List[Dict] = []
    for new_id, (inst, mb) in enumerate(zip(kept_items, kept_masks), start=1):
        fname = f"instance_{new_id:04d}.png"
        mask_path = inst_out / fname
        Image.fromarray((mb.astype(np.uint8) * 255)).save(mask_path)

        bbox, area = safe_bbox_from_mask(mb)

        new_inst = dict(inst)
        new_inst["old_id"] = inst["id"]
        new_inst["id"] = new_id
        new_inst["mask_file"] = f"instances/{fname}"
        new_inst["bbox_xywh"] = bbox
        new_inst["area_px"] = area
        new_instances.append(new_inst)

    # Rebuild metadata
    new_meta = dict(meta)
    new_meta["num_before_dedup"] = len(items)
    new_meta["num_after_dedup"] = len(new_instances)
    new_meta["num_dropped_as_duplicates"] = len(drop)
    new_meta["instances"] = new_instances

    with open(image_out / "instances.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    with open(image_out / "dedup_log.json", "w") as f:
        json.dump(
            {
                "image_file": meta["image_file"],
                "num_before": len(items),
                "num_after": len(new_instances),
                "num_dropped": len(drop),
                "rules": {
                    "iou_thresh": iou_thresh,
                    "contain_thresh": contain_thresh,
                    "bbox_gate": bbox_gate,
                },
                "decisions": decisions,
            },
            f,
            indent=2,
        )

    # Copy files we do want to keep as-is
    for suffix in [
        f"{image_name}_original.png",
        "dropped_masks.json",
        "split_debug.json",
    ]:
        src = image_root / suffix
        if src.exists():
            shutil.copy2(src, image_out / src.name)

    # Rebuild cleaned label image
    if kept_masks:
        label = masks_to_label_image(kept_masks)
    else:
        h, w = rgb.shape[:2]
        label = np.zeros((h, w), dtype=np.uint16)

    label_path = image_out / f"{image_name}_labelIds.png"
    save_label_png(label, label_path)

    # Rebuild cleaned overlay
    overlay_path = image_out / f"{image_name}_overlay.png"
    save_overlay_from_label(rgb, label, overlay_path, alpha=overlay_alpha, seed=123)

    # Rebuild cleaned COCO
    coco: Dict = {
        "images": [
            {
                "id": 1,
                "file_name": meta["image_file"],
                "height": int(label.shape[0]),
                "width": int(label.shape[1]),
            }
        ],
        "categories": [{"id": 1, "name": "particle"}],
        "annotations": [],
    }

    for inst, mb in zip(new_instances, kept_masks):
        bbox = inst.get("bbox_xywh")
        if bbox is None:
            continue

        x, y, w, h = bbox
        coco["annotations"].append(
            {
                "id": int(inst["id"]),
                "image_id": 1,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(inst["area_px"]),
                "segmentation": rle_encode_binary_mask(mb),
                "iscrowd": 0,
                "source": inst.get("source"),
                "parent_candidate_id": inst.get("parent_candidate_id"),
                "old_id": inst.get("old_id"),
            }
        )

    coco_path = image_out / f"{image_name}_coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    return {
        "image_dir": image_root.name,
        "num_before": len(items),
        "num_after": len(new_instances),
        "num_dropped": len(drop),
    }


def process_tarball(tar_path: Path, out_root: Path, **kwargs) -> Dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir_path)

        roots = [p for p in tmpdir_path.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Expected 1 top-level folder in {tar_path}, found {len(roots)}")

        return dedup_one_image(roots[0], out_root, **kwargs)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("tarballs", nargs="+", help="One or more outputs_*.tar.gz files")
    p.add_argument("--out-dir", default="aggregated_clean", help="Output directory")
    p.add_argument("--iou-thresh", type=float, default=0.75)
    p.add_argument("--contain-thresh", type=float, default=0.90)
    p.add_argument("--bbox-gate", type=float, default=0.10)
    p.add_argument("--overlay-alpha", type=float, default=0.43)
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for tar_name in args.tarballs:
        tar_path = Path(tar_name)
        print(f"Processing {tar_path} ...")
        result = process_tarball(
            tar_path,
            out_root=out_root,
            iou_thresh=args.iou_thresh,
            contain_thresh=args.contain_thresh,
            bbox_gate=args.bbox_gate,
            overlay_alpha=args.overlay_alpha,
        )
        summary.append(result)
        print(
            f"  before={result['num_before']} after={result['num_after']} "
            f"dropped={result['num_dropped']}"
        )

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote cleaned outputs to: {out_root}")


if __name__ == "__main__":
    main()