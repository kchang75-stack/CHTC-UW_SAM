# mask_splitter.py
"""
Utilities for splitting likely aggregate / overlapping particle masks.

Main entry point:
    split_aggregate_mask(mask, gray_image, cfg=None, debug=False)

Input:
    mask:      2D bool / 0-1 / 0-255 array for ONE candidate instance
    gray_image: 2D grayscale image (same H,W as mask)

Output:
    List[np.ndarray] of boolean child masks

Method:
    1) clean mask
    2) distance transform inside mask
    3) detect local maxima as seed markers
    4) watershed constrained to the parent mask
    5) return valid child masks

Designed for microscopy-style touching blob separation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np


@dataclass
class SplitConfig:
    # --- pre-cleaning ---
    min_parent_area: int = 80
    open_kernel: int = 0          # 0 disables opening
    close_kernel: int = 3         # fill small gaps / neck noise

    # --- seed detection from distance transform ---
    min_peak_distance: int = 10   # minimum spacing between seed peaks
    peak_rel_threshold: float = 0.35
    peak_abs_threshold: float = 1.5

    # --- child acceptance ---
    min_child_area: int = 40
    max_children: int = 12

    # --- fallback behavior ---
    keep_unsplit_if_no_seeds: bool = True
    keep_unsplit_if_one_child: bool = True

    # --- watershed ---
    watershed_dilate_iter: int = 1


def _to_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask to a clean boolean mask."""
    if mask.dtype == bool:
        mb = mask.copy()
    else:
        mb = mask > 0
    return mb.astype(bool)


def _kernel(k: int) -> np.ndarray | None:
    if k is None or k <= 1:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _clean_mask(mask_bool: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """Apply optional morphology cleanup."""
    m = (mask_bool.astype(np.uint8) * 255)

    k_open = _kernel(cfg.open_kernel)
    if k_open is not None:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    k_close = _kernel(cfg.close_kernel)
    if k_close is not None:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    return m > 0


def _distance_transform(mask_bool: np.ndarray) -> np.ndarray:
    m = (mask_bool.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return dist


def _find_seed_markers(
    dist: np.ndarray,
    mask_bool: np.ndarray,
    cfg: SplitConfig,
) -> Tuple[np.ndarray, int]:
    """
    Find local maxima in the distance map and convert them into connected markers.

    Returns:
        markers_int32, n_markers
    """
    if dist.size == 0 or dist.max() <= 0:
        markers = np.zeros(dist.shape, dtype=np.int32)
        return markers, 0

    # Threshold candidate peaks.
    rel_thr = cfg.peak_rel_threshold * float(dist.max())
    thr = max(cfg.peak_abs_threshold, rel_thr)

    peak_mask = dist >= thr
    peak_mask &= mask_bool

    # Non-maximum suppression via dilation.
    # A pixel is a peak if it equals the local dilated max and exceeds threshold.
    k = max(1, int(cfg.min_peak_distance))
    # Use odd kernel size for local maximum neighborhood.
    ksize = 2 * k + 1
    nms_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dist_dil = cv2.dilate(dist, nms_kernel)
    local_max = (dist == dist_dil) & peak_mask

    # Convert peaks into connected components for watershed markers.
    local_max_u8 = local_max.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(local_max_u8)

    # labels: 0 background, 1..n_labels-1 markers
    markers = labels.astype(np.int32)
    n_markers = n_labels - 1

    return markers, n_markers


def _watershed_split(
    mask_bool: np.ndarray,
    gray_image: np.ndarray,
    markers: np.ndarray,
    cfg: SplitConfig,
) -> np.ndarray:
    """
    Run watershed constrained to the parent mask.

    Returns:
        label image where 0=background/outside, -1=boundary, >0 child labels
    """
    h, w = mask_bool.shape

    # Watershed needs a 3-channel image.
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be 2D")
    ws_img = cv2.cvtColor(gray_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Build watershed markers:
    # - outside mask should be 0 / unknown
    # - seed markers are 1..N
    ws_markers = np.zeros((h, w), dtype=np.int32)
    ws_markers[markers > 0] = markers[markers > 0]

    # Give the whole mask an "unknown but allowed" region, while leaving
    # seed labels intact.
    # A slight dilation can help avoid unstable boundaries.
    parent_u8 = mask_bool.astype(np.uint8)
    if cfg.watershed_dilate_iter > 0:
        parent_u8 = cv2.dilate(parent_u8, _kernel(3), iterations=cfg.watershed_dilate_iter)

    parent_u8 = (parent_u8 > 0).astype(np.uint8)
    ws_markers[(parent_u8 > 0) & (ws_markers == 0)] = 0

    # To prevent watershed from expanding outside the original mask, brighten
    # outside-mask pixels to a uniform value so boundaries tend to stay inside.
    ws_img = ws_img.copy()
    ws_img[~mask_bool] = 0

    cv2.watershed(ws_img, ws_markers)

    # Remove anything outside original mask.
    ws_markers[~mask_bool] = 0
    return ws_markers


def _extract_child_masks(
    labels: np.ndarray,
    cfg: SplitConfig,
) -> List[np.ndarray]:
    """Convert watershed labels into boolean child masks."""
    child_masks: List[np.ndarray] = []

    unique_labels = [x for x in np.unique(labels) if x > 0]
    for lab in unique_labels:
        cm = labels == lab
        area = int(cm.sum())
        if area < cfg.min_child_area:
            continue
        child_masks.append(cm)

    # Sort largest first for consistency.
    child_masks.sort(key=lambda m: int(m.sum()), reverse=True)

    if len(child_masks) > cfg.max_children:
        child_masks = child_masks[: cfg.max_children]

    return child_masks


def split_aggregate_mask(
    mask: np.ndarray,
    gray_image: np.ndarray,
    cfg: SplitConfig | None = None,
    debug: bool = False,
) -> List[np.ndarray]:
    """
    Split a likely aggregate mask into child masks using distance-transform seeds + watershed.

    Returns:
        list of boolean masks
    """
    if cfg is None:
        cfg = SplitConfig()

    mask_bool = _to_bool_mask(mask)

    if gray_image.shape != mask_bool.shape:
        raise ValueError(
            f"gray_image shape {gray_image.shape} does not match mask shape {mask_bool.shape}"
        )

    parent_area = int(mask_bool.sum())
    if parent_area < cfg.min_parent_area:
        if debug:
            print(f"[split] parent too small to split: area={parent_area}")
        return [mask_bool] if cfg.keep_unsplit_if_no_seeds else []

    mask_bool = _clean_mask(mask_bool, cfg)
    parent_area = int(mask_bool.sum())

    dist = _distance_transform(mask_bool)
    markers, n_markers = _find_seed_markers(dist, mask_bool, cfg)

    if debug:
        print(
            f"[split] area={parent_area}, dist_max={float(dist.max()):.3f}, "
            f"n_markers={n_markers}"
        )

    if n_markers <= 0:
        return [mask_bool] if cfg.keep_unsplit_if_no_seeds else []

    if n_markers == 1:
        return [mask_bool] if cfg.keep_unsplit_if_one_child else []

    labels = _watershed_split(mask_bool, gray_image, markers, cfg)
    child_masks = _extract_child_masks(labels, cfg)

    if debug:
        child_areas = [int(cm.sum()) for cm in child_masks]
        print(f"[split] children_kept={len(child_masks)}, child_areas={child_areas}")

    if len(child_masks) <= 1:
        return [mask_bool] if cfg.keep_unsplit_if_one_child else []

    return child_masks


def split_aggregate_mask_with_debug(
    mask: np.ndarray,
    gray_image: np.ndarray,
    cfg: SplitConfig | None = None,
) -> Dict[str, Any]:
    """
    Same as split_aggregate_mask, but returns debug artifacts too.
    Useful when you want to save intermediate images.
    """
    if cfg is None:
        cfg = SplitConfig()

    mask_bool = _to_bool_mask(mask)
    if gray_image.shape != mask_bool.shape:
        raise ValueError(
            f"gray_image shape {gray_image.shape} does not match mask shape {mask_bool.shape}"
        )

    cleaned = _clean_mask(mask_bool, cfg)
    dist = _distance_transform(cleaned)
    markers, n_markers = _find_seed_markers(dist, cleaned, cfg)

    if n_markers <= 0:
        children = [cleaned] if cfg.keep_unsplit_if_no_seeds else []
        labels = np.zeros_like(markers, dtype=np.int32)
    elif n_markers == 1:
        children = [cleaned] if cfg.keep_unsplit_if_one_child else []
        labels = np.zeros_like(markers, dtype=np.int32)
    else:
        labels = _watershed_split(cleaned, gray_image, markers, cfg)
        children = _extract_child_masks(labels, cfg)
        if len(children) <= 1 and cfg.keep_unsplit_if_one_child:
            children = [cleaned]

    return {
        "cleaned_mask": cleaned,
        "distance": dist,
        "markers": markers,
        "labels": labels,
        "children": children,
        "n_markers": int(n_markers),
    }