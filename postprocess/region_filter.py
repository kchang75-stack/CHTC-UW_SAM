# postprocess/region_filter.py

from dataclasses import dataclass
from typing import List, Tuple

from region_features import RegionFeatures


# ----------------------------------------------------------------------
# 1. Threshold configuration
# ----------------------------------------------------------------------

@dataclass
class FilterConfig:
    """
    All the knobs you can tune for rule-based filtering.
    Defaults are pass-through (no filtering) until you tune them.
    """
    # --- size-related ---
    min_area: float = 20.0
    max_area: float = 1e9

    # --- shape-related ---
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    max_aspect_ratio: float = 10.0
    min_solidity: float = 0.0

    # --- intensity-related ---
    min_mean_intensity: float = 0.0
    max_mean_intensity: float = 255.0
    max_edge_intensity_ratio: float = 999.0


# ----------------------------------------------------------------------
# 2. Single-region decision function
# ----------------------------------------------------------------------

def decide_keep_region(
    feats: RegionFeatures,
    cfg: FilterConfig,
    debug: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Decide whether to keep this region based on its features and cfg.
    Returns (keep, reasons) where reasons lists why it was dropped.
    """
    reasons: List[str] = []

    # --- 1) Size checks ------------------------------------------------
    if feats.area < cfg.min_area:
        reasons.append(f"area {feats.area:.1f} < min_area {cfg.min_area:.1f}")
    if feats.area > cfg.max_area:
        reasons.append(f"area {feats.area:.1f} > max_area {cfg.max_area:.1f}")

    # --- 2) Shape checks -----------------------------------------------
    if feats.circularity < cfg.min_circularity:
        reasons.append(f"circularity {feats.circularity:.2f} < min_circularity {cfg.min_circularity:.2f}")
    if feats.circularity > cfg.max_circularity:
        reasons.append(f"circularity {feats.circularity:.2f} > max_circularity {cfg.max_circularity:.2f}")
    if feats.aspect_ratio > cfg.max_aspect_ratio:
        reasons.append(f"aspect_ratio {feats.aspect_ratio:.2f} > max_aspect_ratio {cfg.max_aspect_ratio:.2f}")
    if feats.solidity < cfg.min_solidity:
        reasons.append(f"solidity {feats.solidity:.2f} < min_solidity {cfg.min_solidity:.2f}")

    # --- 3) Intensity checks -------------------------------------------
    if feats.mean_intensity < cfg.min_mean_intensity:
        reasons.append(f"mean_intensity {feats.mean_intensity:.1f} < min_mean_intensity {cfg.min_mean_intensity:.1f}")
    if feats.mean_intensity > cfg.max_mean_intensity:
        reasons.append(f"mean_intensity {feats.mean_intensity:.1f} > max_mean_intensity {cfg.max_mean_intensity:.1f}")
    if feats.edge_intensity_ratio > cfg.max_edge_intensity_ratio:
        reasons.append(f"edge_intensity_ratio {feats.edge_intensity_ratio:.2f} > max_edge_intensity_ratio {cfg.max_edge_intensity_ratio:.2f}")

    keep = len(reasons) == 0

    if debug:
        if keep:
            print(f"[KEEP] idx={feats.idx}")
        else:
            print(f"[DROP] idx={feats.idx} | " + " ; ".join(reasons))

    return keep, reasons


# ----------------------------------------------------------------------
# 3. Apply to all regions
# ----------------------------------------------------------------------

def filter_regions(
    all_feats: List[RegionFeatures],
    cfg: FilterConfig,
    debug: bool = False,
) -> List[int]:
    """
    Returns list of indices of regions to keep.
    """
    keep_indices: List[int] = []
    for feats in all_feats:
        keep, _ = decide_keep_region(feats, cfg, debug=debug)
        if keep:
            keep_indices.append(feats.idx)
    return keep_indices