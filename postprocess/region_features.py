# postprocess/region_features.py

from dataclasses import dataclass
import math
import numpy as np
import cv2


@dataclass
class RegionFeatures:
    idx: int
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float
    mean_intensity: float
    std_intensity: float
    edge_intensity_ratio: float
    bbox_width: int
    bbox_height: int


def compute_region_features_for_all(rgb, masks):
    import cv2
    feats = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for idx, m in enumerate(masks):
        seg = m["segmentation"].astype(bool)
        seg_uint8 = (seg.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cont = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cont)
        perimeter = cv2.arcLength(cont, closed=True)
        circularity = float(4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(cont)
        short_side = min(w, h)
        long_side = max(w, h)
        aspect_ratio = float(long_side / short_side) if short_side > 0 else 1.0
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        pix_vals = gray[seg]
        mean_intensity = float(pix_vals.mean()) if pix_vals.size > 0 else 0.0
        std_intensity = float(pix_vals.std()) if pix_vals.size > 0 else 0.0
        dist = cv2.distanceTransform(seg_uint8, cv2.DIST_L2, 3)
        max_dist = dist.max()
        edge_intensity_ratio = 1.0
        if max_dist > 0:
            edge_mask = (dist > 0) & (dist <= 0.25 * max_dist)
            center_mask = dist >= 0.50 * max_dist
            edge_vals = gray[edge_mask]
            center_vals = gray[center_mask]
            if edge_vals.size > 0 and center_vals.size > 0:
                center_mean = float(center_vals.mean())
                if center_mean > 1e-6:
                    edge_intensity_ratio = float(edge_vals.mean()) / center_mean
        feats.append(RegionFeatures(
            idx=idx,
            area=float(area),
            perimeter=float(perimeter),
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            edge_intensity_ratio=edge_intensity_ratio,
            bbox_width=int(w),
            bbox_height=int(h),
        ))
    return feats
