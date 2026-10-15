#!/usr/bin/env python3
"""
SAM3 inference on one image (CHTC-friendly) with keep / split / drop postprocessing.

Outputs (under --out-dir):
  <stem>_original.png
  <stem>_labelIds.png      uint16 label image (0=bg, 1..N instances)
  <stem>_overlay.png       coloured overlay blended with original
  instances/instance_####.png
  instances.json
  <stem>_coco.json
  dropped_masks.json
  split_debug.json
  <outputs>.tar.gz

Example:
  python3 sam3_test.py \
      --image input.jpg \
      --prompt particle \
      --threshold 0.05 \
      --mask-threshold 0.10 \
      --alpha 110
"""

# -----------------------------------------------------------------------
# MUST COME FIRST: set writable cache dirs before any HF/torch imports
# -----------------------------------------------------------------------
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"DEBUG script_dir: {_script_dir}", flush=True)
print(f"DEBUG contents: {os.listdir(_script_dir)}", flush=True)
if _script_dir not in sys.path:
    sys.path.append(_script_dir)

scratch = os.environ.get("_CONDOR_SCRATCH_DIR") or os.getcwd()
cache_root = os.path.join(scratch, "hf_cache")
tmp_root = os.path.join(scratch, "tmp")
os.makedirs(cache_root, exist_ok=True)
os.makedirs(tmp_root, exist_ok=True)

os.environ["HOME"] = scratch
os.environ["XDG_CACHE_HOME"] = cache_root
os.environ["XDG_CONFIG_HOME"] = os.path.join(scratch, "xdg_config")
os.environ["XDG_DATA_HOME"] = os.path.join(scratch, "xdg_data")
os.environ["TMPDIR"] = tmp_root
os.environ["TMP"] = tmp_root
os.environ["TEMP"] = tmp_root

os.environ["HF_HOME"] = cache_root
os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")

os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# -----------------------------------------------------------------------
# Standard imports
# -----------------------------------------------------------------------
import argparse
import json
import tarfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# -----------------------------------------------------------------------
# Postprocess imports
# -----------------------------------------------------------------------
from region_features import compute_region_features_for_all
from region_filter import FilterConfig, decide_keep_region
from mask_splitter import SplitConfig, split_aggregate_mask


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # SAM / I/O
    p.add_argument("--image", default="input.jpg")
    p.add_argument("--prompt", default="object")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--mask-threshold", type=float, default=0.10)
    p.add_argument("--alpha", type=int, default=110)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--tar", default="outputs.tar.gz")

    # Filter debug
    p.add_argument("--filter-debug", action="store_true")

    return p.parse_args()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def read_hf_token() -> Optional[str]:
    path = os.path.join(os.getcwd(), "hf_token.txt")
    if os.path.exists(path):
        tok = open(path).read().strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    return None


def mask_bbox_area(mask_255: np.ndarray) -> Tuple[Optional[List[int]], int]:
    ys, xs = np.where(mask_255 > 0)
    if xs.size == 0:
        return None, 0
    return (
        [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) - int(xs.min()) + 1,
            int(ys.max()) - int(ys.min()) + 1,
        ],
        int(xs.size),
    )


def rle_encode(mask: np.ndarray) -> Dict:
    pixels = mask.T.flatten().astype(np.uint8)
    counts, prev, run = [], 0, 0
    for p in pixels:
        p = int(p)
        if p == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = p
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def masks_to_label(bin_masks: List[np.ndarray]) -> Optional[np.ndarray]:
    if not bin_masks:
        return None
    h, w = bin_masks[0].shape
    label = np.zeros((h, w), dtype=np.uint32)
    for new_id, i in enumerate(
        sorted(range(len(bin_masks)), key=lambda i: int(bin_masks[i].sum()), reverse=True),
        start=1,
    ):
        label[bin_masks[i]] = new_id
    return label.astype(np.uint16)


def save_overlay(
    image_rgb: Image.Image,
    label: np.ndarray,
    path: str,
    alpha: float,
    seed: int = 123,
) -> None:
    rgb = np.array(image_rgb, dtype=np.uint8)
    lbl = label.astype(np.int32)
    max_id = int(lbl.max())
    if max_id == 0:
        image_rgb.save(path)
        return
    rng = np.random.default_rng(seed)
    cmap = np.zeros((max_id + 1, 3), dtype=np.uint8)
    cmap[1:] = rng.integers(0, 256, size=(max_id, 3), dtype=np.uint8)
    overlay = (rgb * (1 - alpha) + cmap[lbl] * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def feature_dict(feats, one_based_id: int, extra: Optional[Dict] = None) -> Dict:
    d = {
        "id": one_based_id,
        "area_px": float(feats.area),
        "circularity": round(float(feats.circularity), 4),
        "aspect_ratio": round(float(feats.aspect_ratio), 4),
        "solidity": round(float(feats.solidity), 4),
        "mean_intensity": round(float(feats.mean_intensity), 2),
        "edge_intensity_ratio": round(float(feats.edge_intensity_ratio), 4),
    }
    if extra:
        d.update(extra)
    return d


def is_likely_aggregate(feats) -> Tuple[bool, List[str]]:
    """
    Heuristic to decide whether a mask should be SPLIT instead of directly kept/dropped.
    Tune these based on your microscopy images.
    """
    reasons: List[str] = []

    if feats.area > 8000:
        reasons.append(f"large_area {feats.area:.1f} > 8000")

    if feats.circularity < 0.60 and feats.area > 1500:
        reasons.append(
            f"low_circularity_with_area circularity={feats.circularity:.2f}, area={feats.area:.1f}"
        )

    if feats.aspect_ratio > 2.0:
        reasons.append(f"elongated aspect_ratio {feats.aspect_ratio:.2f} > 2.0")

    return (len(reasons) > 0), reasons


def keep_split_child(feats, child_cfg: FilterConfig) -> Tuple[bool, List[str]]:
    return decide_keep_region(feats, child_cfg, debug=False)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    hf_token = read_hf_token()

    print("CWD:", os.getcwd())
    print("CONDOR_SCRATCH:", os.environ.get("_CONDOR_SCRATCH_DIR"))
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
    print("HOME:", os.environ.get("HOME"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha01 = max(0, min(255, args.alpha)) / 255.0

    print(f"Prompt: {args.prompt!r}")
    print(f"threshold={args.threshold}, mask_threshold={args.mask_threshold}")
    print(f"Image: {args.image}")
    print("HF token present:", bool(hf_token))

    # ------------------------------------------------------------
    # Hard-coded configs for tuning
    # ------------------------------------------------------------
    keep_cfg = FilterConfig(
        min_area=1000.0,
        max_area=100000.0,
        min_circularity=0.65,
        max_circularity=1.0,
        max_aspect_ratio=2.0,
        min_solidity=0.4,
        min_mean_intensity=80.0,
        max_mean_intensity=255.0,
        max_edge_intensity_ratio=1.40,
    )

    child_cfg = FilterConfig(
        min_area=40.0,
        max_area=100000.0,
        min_circularity=0.75,
        max_circularity=1.0,
        max_aspect_ratio=3.0,
        min_solidity=0.2,
        min_mean_intensity=60.0,
        max_mean_intensity=255.0,
        max_edge_intensity_ratio=1.5,
    )

    split_cfg = SplitConfig(
        min_parent_area=300,
        open_kernel=0,
        close_kernel=3,
        min_peak_distance=10,
        peak_rel_threshold=0.30,
        peak_abs_threshold=1.5,
        min_child_area=60,
        max_children=8,
        keep_unsplit_if_no_seeds=True,
        keep_unsplit_if_one_child=True,
        watershed_dilate_iter=1,
    )

    print("Keep config:", keep_cfg)
    print("Child config:", child_cfg)
    print("Split config:", split_cfg)

    # ------------------------------------------------------------
    # Load model / processor
    # ------------------------------------------------------------
    print("Loading model...")
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)

    print("Loading processor...")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    print("Opening image...")
    image = Image.open(args.image).convert("RGB")
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------
    # SAM inference
    # ------------------------------------------------------------
    print("Preparing inputs...")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    print("=== DEBUG OUTPUT STRUCTURE ===")
    print("type(outputs):", type(outputs))

    if hasattr(outputs, "keys"):
        print("outputs keys:", outputs.keys())
    else:
        print("outputs:", outputs)

    # Also inspect masks specifically
    try:
        print("pred_masks shape:", outputs.pred_masks.shape)
    except Exception as e:
        print("No pred_masks found:", e)
    print("Post-processing...")
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    raw_masks = results["masks"]
    n_raw = int(len(raw_masks))
    print(f"num masks (before filtering): {n_raw}")

    # Convert raw masks to bool masks
    candidate_masks: List[np.ndarray] = []
    for i in range(n_raw):
        m = raw_masks[i].detach().cpu().numpy()
        mb = m > args.mask_threshold
        if mb.ndim != 2:
            raise ValueError(f"Mask {i} shape {mb.shape}, expected (H,W)")
        if int(mb.sum()) == 0:
            continue
        candidate_masks.append(mb.astype(bool))

    print(f"num valid candidate masks: {len(candidate_masks)}")

    # ------------------------------------------------------------
    # Compute features for raw candidates
    # ------------------------------------------------------------
    sam1_masks = [{"segmentation": mb} for mb in candidate_masks]

    print("Computing region features for candidates...")
    all_feats = compute_region_features_for_all(rgb, sam1_masks)

    # ------------------------------------------------------------
    # keep / split / drop
    # ------------------------------------------------------------
    print("Running keep / split / drop pipeline...")

    final_items: List[Dict] = []
    dropped_log: List[Dict] = []
    split_log: List[Dict] = []

    for feats in all_feats:
        parent_idx = feats.idx
        parent_mask = candidate_masks[parent_idx]

        # # 1) decide whether this looks like an aggregate
        do_split, split_reasons = is_likely_aggregate(feats)

        # # 2) if likely aggregate -> try split first
        # if do_split:
        #     if args.filter_debug:
        #         print(f"[SPLIT-TRY] idx={parent_idx} reasons={split_reasons}")

        #     child_masks = split_aggregate_mask(
        #         mask=parent_mask,
        #         gray_image=gray,
        #         cfg=split_cfg,
        #         debug=args.filter_debug,
        #     )

        #     if len(child_masks) > 1:
        #         child_feats_all = compute_region_features_for_all(
        #             rgb, [{"segmentation": cm} for cm in child_masks]
        #         )

        #         kept_children = 0
        #         dropped_children = 0
        #         child_ids_kept: List[int] = []

        #         for child_local_idx, child_feats in enumerate(child_feats_all):
        #             keep_child, child_reasons = keep_split_child(child_feats, child_cfg)

        #             if keep_child:
        #                 final_items.append(
        #                     {
        #                         "mask": child_masks[child_feats.idx],
        #                         "source": "split_child_kept",
        #                         "parent_candidate_id": parent_idx + 1,
        #                         "split_reasons": split_reasons,
        #                     }
        #                 )
        #                 kept_children += 1
        #                 child_ids_kept.append(child_local_idx + 1)
        #             else:
        #                 dropped_children += 1
        #                 dropped_log.append(
        #                     feature_dict(
        #                         child_feats,
        #                         one_based_id=child_local_idx + 1,
        #                         extra={
        #                             "decision": "drop_split_child",
        #                             "parent_candidate_id": parent_idx + 1,
        #                             "reasons": child_reasons,
        #                         },
        #                     )
        #                 )

        #         split_log.append(
        #             {
        #                 "parent_candidate_id": parent_idx + 1,
        #                 "parent_features": feature_dict(
        #                     feats,
        #                     one_based_id=parent_idx + 1,
        #                     extra={"split_trigger_reasons": split_reasons},
        #                 ),
        #                 "num_children_produced": len(child_masks),
        #                 "num_children_kept": kept_children,
        #                 "num_children_dropped": dropped_children,
        #                 "child_ids_kept": child_ids_kept,
        #             }
        #         )

        #         if kept_children > 0:
        #             continue
        #         else:
        #             # fallback: if split yielded no acceptable children, try keep/drop parent
        #             keep_parent, keep_reasons = decide_keep_region(
        #                 feats, keep_cfg, debug=args.filter_debug
        #             )
        #             if keep_parent:
        #                 final_items.append(
        #                     {
        #                         "mask": parent_mask,
        #                         "source": "kept_parent_after_failed_child_filter",
        #                         "parent_candidate_id": None,
        #                         "split_reasons": split_reasons,
        #                     }
        #                 )
        #             else:
        #                 dropped_log.append(
        #                     feature_dict(
        #                         feats,
        #                         one_based_id=parent_idx + 1,
        #                         extra={
        #                             "decision": "drop_parent_after_failed_split_children",
        #                             "reasons": keep_reasons,
        #                             "split_reasons": split_reasons,
        #                         },
        #                     )
        #                 )
        #             continue

        #     # split attempt produced <=1 child -> fall through to keep/drop parent

        # 3) normal keep/drop for non-aggregate or unsplit parent
        keep_parent, reasons = decide_keep_region(feats, keep_cfg, debug=args.filter_debug)

        if keep_parent:
            final_items.append(
                {
                    "mask": parent_mask,
                    "source": "kept_single",
                    "parent_candidate_id": None,
                    "split_reasons": split_reasons if do_split else [],
                }
            )
        else:
            dropped_log.append(
                feature_dict(
                    feats,
                    one_based_id=parent_idx + 1,
                    extra={
                        "decision": "drop_parent",
                        "reasons": reasons,
                        "split_reasons": split_reasons if do_split else [],
                    },
                )
            )

    # Final masks
    bin_masks = [item["mask"] for item in final_items]
    n_kept = len(bin_masks)

    print(f"Kept {n_kept} / {len(candidate_masks)} candidate masks after keep/split/drop")

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------
    out_dir = args.out_dir
    inst_dir = os.path.join(out_dir, "instances")
    os.makedirs(inst_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    # Save original
    orig_path = os.path.join(out_dir, f"{stem}_original.png")
    image.save(orig_path)

    # ------------------------------------------------------------
    # Save per-instance masks and metadata
    # ------------------------------------------------------------
    meta: Dict = {
        "image_file": args.image,
        "prompt": args.prompt,
        "threshold": float(args.threshold),
        "mask_threshold": float(args.mask_threshold),
        "num_raw_masks": n_raw,
        "num_candidate_masks": len(candidate_masks),
        "num_final_instances": n_kept,
        "keep_config": {
            "min_area": keep_cfg.min_area,
            "max_area": keep_cfg.max_area,
            "min_circularity": keep_cfg.min_circularity,
            "max_circularity": keep_cfg.max_circularity,
            "max_aspect_ratio": keep_cfg.max_aspect_ratio,
            "min_solidity": keep_cfg.min_solidity,
            "min_mean_intensity": keep_cfg.min_mean_intensity,
            "max_mean_intensity": keep_cfg.max_mean_intensity,
            "max_edge_intensity_ratio": keep_cfg.max_edge_intensity_ratio,
        },
        "child_config": {
            "min_area": child_cfg.min_area,
            "max_area": child_cfg.max_area,
            "min_circularity": child_cfg.min_circularity,
            "max_circularity": child_cfg.max_circularity,
            "max_aspect_ratio": child_cfg.max_aspect_ratio,
            "min_solidity": child_cfg.min_solidity,
            "min_mean_intensity": child_cfg.min_mean_intensity,
            "max_mean_intensity": child_cfg.max_mean_intensity,
            "max_edge_intensity_ratio": child_cfg.max_edge_intensity_ratio,
        },
        "split_config": {
            "min_parent_area": split_cfg.min_parent_area,
            "open_kernel": split_cfg.open_kernel,
            "close_kernel": split_cfg.close_kernel,
            "min_peak_distance": split_cfg.min_peak_distance,
            "peak_rel_threshold": split_cfg.peak_rel_threshold,
            "peak_abs_threshold": split_cfg.peak_abs_threshold,
            "min_child_area": split_cfg.min_child_area,
            "max_children": split_cfg.max_children,
            "keep_unsplit_if_no_seeds": split_cfg.keep_unsplit_if_no_seeds,
            "keep_unsplit_if_one_child": split_cfg.keep_unsplit_if_one_child,
            "watershed_dilate_iter": split_cfg.watershed_dilate_iter,
        },
        "instances": [],
    }

    for inst_id, item in enumerate(final_items, start=1):
        mb = item["mask"]
        m255 = mb.astype(np.uint8) * 255
        bbox, area = mask_bbox_area(m255)
        fname = f"instance_{inst_id:04d}.png"
        Image.fromarray(m255).save(os.path.join(inst_dir, fname))

        meta["instances"].append(
            {
                "id": inst_id,
                "mask_file": f"instances/{fname}",
                "bbox_xywh": bbox,
                "area_px": int(area),
                "source": item["source"],
                "parent_candidate_id": item["parent_candidate_id"],
                "split_reasons": item["split_reasons"],
            }
        )

    with open(os.path.join(out_dir, "instances.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------
    # Save drop and split logs
    # ------------------------------------------------------------
    dropped_path = os.path.join(out_dir, "dropped_masks.json")
    with open(dropped_path, "w") as f:
        json.dump(
            {
                "image_file": args.image,
                "total_raw_masks": n_raw,
                "total_candidate_masks": len(candidate_masks),
                "total_final_instances": n_kept,
                "total_dropped_entries": len(dropped_log),
                "keep_config": meta["keep_config"],
                "child_config": meta["child_config"],
                "split_config": meta["split_config"],
                "dropped": dropped_log,
            },
            f,
            indent=2,
        )
    print(f"Wrote {dropped_path}")

    split_debug_path = os.path.join(out_dir, "split_debug.json")
    with open(split_debug_path, "w") as f:
        json.dump(
            {
                "image_file": args.image,
                "num_split_events": len(split_log),
                "split_events": split_log,
            },
            f,
            indent=2,
        )
    print(f"Wrote {split_debug_path}")

    # ------------------------------------------------------------
    # Save label image and overlay
    # ------------------------------------------------------------
    label = masks_to_label(bin_masks)
    if label is None:
        label = np.zeros((image.height, image.width), dtype=np.uint16)

    label_path = os.path.join(out_dir, f"{stem}_labelIds.png")
    Image.fromarray(label, mode="I;16").save(label_path)
    print(f"Wrote label IDs {label_path}")

    overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
    save_overlay(image, label, overlay_path, alpha=alpha01)
    print(f"Wrote {overlay_path}")

    # ------------------------------------------------------------
    # Save COCO-ish annotations
    # ------------------------------------------------------------
    coco: Dict = {
        "images": [
            {
                "id": 1,
                "file_name": os.path.basename(args.image),
                "height": int(label.shape[0]),
                "width": int(label.shape[1]),
            }
        ],
        "categories": [{"id": 1, "name": "particle"}],
        "annotations": [],
    }

    for inst in meta["instances"]:
        inst_id = inst["id"]
        if inst["bbox_xywh"] is None:
            continue
        x, y, w, h = inst["bbox_xywh"]
        coco["annotations"].append(
            {
                "id": inst_id,
                "image_id": 1,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(inst["area_px"]),
                "segmentation": rle_encode(bin_masks[inst_id - 1]),
                "iscrowd": 0,
                "source": inst["source"],
                "parent_candidate_id": inst["parent_candidate_id"],
            }
        )

    coco_path = os.path.join(out_dir, f"{stem}_coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Wrote COCO {coco_path}")

    print(f"Wrote {orig_path}")
    print(f"Wrote {n_kept} masks under {inst_dir}/ and metadata {out_dir}/instances.json")

    # ------------------------------------------------------------
    # Optional tarball
    # ------------------------------------------------------------
    if args.tar:
        print(f"Packing outputs into {args.tar} ...")
        with tarfile.open(args.tar, "w:gz") as tar:
            tar.add(out_dir, arcname=os.path.basename(out_dir))
        print(f"Wrote {args.tar}")

    print("Done.")


if __name__ == "__main__":
    main()