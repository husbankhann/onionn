"""
Onion Quality Assessment & Grading Engine
==========================================
Converts subjective visual inspection into repeatable, quantitative
measurements so that grading is consistent across procurement centers.

Pipeline per onion image:
  1. Segment onion(s) from background
  2. Measure size (equivalent diameter, area) in mm using a reference
     object OR a calibrated pixels-per-mm value for the camera rig
  3. Measure shape (roundness / eccentricity)
  4. Measure surface defects (blemishes, sprouting, mold, rot, cuts)
     via color-anomaly + texture analysis
  5. Measure skin color uniformity (for variety/ripeness consistency)
  6. Combine sub-scores into a single composite quality score
  7. Map composite score -> standard grade (A/B/C/Reject) using
     configurable, auditable thresholds (not a black box)

Every measurement is stored so a human inspector can review *why*
a grade was assigned (auditability is what resolves disputes).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration — these thresholds should be tunable per procurement center
# via the /api/config endpoint, and every grading run stores which config
# version was used, so disputes can be traced to an exact rule set.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "config_version": "1.0",
    # Size grading bands in millimetres (equivalent diameter)
    "size_bands_mm": {
        "small": [0, 40],
        "medium": [40, 60],
        "large": [60, 80],
        "extra_large": [80, 999],
    },
    # Composite score thresholds -> grade
    "grade_thresholds": {
        "A": 85,   # >= 85
        "B": 70,   # >= 70
        "C": 50,   # >= 50
        # below 50 -> "Reject"
    },
    # Weights for composite score (must sum to 100)
    "weights": {
        "defect_score": 40,
        "shape_score": 20,
        "color_uniformity_score": 20,
        "size_conformity_score": 20,
    },
    # Defect detection sensitivity
    "defect_thresholds": {
        "dark_spot_ratio_max": 0.15,     # fraction of onion area that can be dark/rot before penalty
        "sprout_green_ratio_max": 0.03,  # green sprouting fraction
        "mold_saturation_drop": 25,      # saturation delta indicating mold/dry patches
    },
    # Camera calibration: pixels per millimetre for the fixed procurement-center rig.
    # If a reference coin/marker is in frame instead, calibration is computed per-image.
    "pixels_per_mm": 8.0,
}


@dataclass
class OnionMeasurement:
    onion_id: int
    equivalent_diameter_mm: float
    area_mm2: float
    roundness: float                # 1.0 = perfect circle
    eccentricity: float
    defect_area_ratio: float
    dark_spot_ratio: float
    sprout_ratio: float
    color_uniformity_score: float
    size_band: str
    size_conformity_score: float
    shape_score: float
    defect_score: float
    composite_score: float
    grade: str
    flags: list = field(default_factory=list)   # human-readable reasons, for dispute resolution

    def to_dict(self):
        return asdict(self)


class OnionGradingEngine:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _segment_onions(self, image_bgr: np.ndarray):
        """
        Segments onion(s) from a (typically plain/tray) background.
        Returns list of (contour, mask) for each detected onion.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Background is assumed to be a fairly uniform tray/mat.
        # Use Otsu threshold on saturation + value combined with edge info
        # to be robust to brown/red/yellow/white onion varieties.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        _, sat_thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, val_thresh = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        combined = cv2.bitwise_or(sat_thresh, cv2.bitwise_not(val_thresh))
        combined = cv2.medianBlur(combined, 9)

        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 0.005 * image_bgr.shape[0] * image_bgr.shape[1]
        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            results.append((c, mask))

        # Largest-first so onion_id ordering is stable/deterministic
        results.sort(key=lambda cm: cv2.contourArea(cm[0]), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-onion measurements
    # ------------------------------------------------------------------
    def _measure_shape(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0, 1.0, 0.0

        # Roundness: 4*pi*Area / Perimeter^2  (1.0 = perfect circle)
        roundness = float(4 * np.pi * area / (perimeter ** 2))
        roundness = min(roundness, 1.0)

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (major, minor), _ = ellipse
            major, minor = max(major, minor), min(major, minor)
            eccentricity = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0
        else:
            eccentricity = 0.0

        equivalent_diameter_px = float(np.sqrt(4 * area / np.pi))
        return roundness, eccentricity, equivalent_diameter_px

    def _measure_defects(self, image_bgr, mask):
        """
        Detects dark spots (rot/bruising), sprouting (green shoots),
        and mold/dry patches (localized saturation collapse) within
        the onion's own mask.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        onion_pixels_mask = mask > 0
        total_px = int(np.sum(onion_pixels_mask))
        if total_px == 0:
            return 1.0, 1.0, 0.0, ["segmentation_failed"]

        h, s, v = cv2.split(hsv)

        # Dark spot / rot detection: low value (brightness) relative to
        # the onion's own median brightness (adapts to lighting + variety).
        median_v = np.median(v[onion_pixels_mask])
        dark_mask = (v < (median_v * 0.55)) & onion_pixels_mask
        dark_ratio = float(np.sum(dark_mask)) / total_px

        # Sprouting: green hue range (35-85 in OpenCV's 0-179 H scale)
        green_mask = (h > 35) & (h < 85) & (s > 40) & onion_pixels_mask
        sprout_ratio = float(np.sum(green_mask)) / total_px

        # Mold / dry patch: localized saturation drop vs. median saturation
        median_s = np.median(s[onion_pixels_mask])
        mold_mask = (s < max(median_s - self.config["defect_thresholds"]["mold_saturation_drop"], 0)) & onion_pixels_mask
        mold_ratio = float(np.sum(mold_mask)) / total_px

        flags = []
        thr = self.config["defect_thresholds"]
        if dark_ratio > thr["dark_spot_ratio_max"]:
            flags.append(f"excess_dark_spots_{dark_ratio:.2%}")
        if sprout_ratio > thr["sprout_green_ratio_max"]:
            flags.append(f"sprouting_detected_{sprout_ratio:.2%}")
        if mold_ratio > 0.10:
            flags.append(f"possible_mold_or_dry_patch_{mold_ratio:.2%}")

        total_defect_ratio = min(dark_ratio + sprout_ratio + mold_ratio, 1.0)
        return dark_ratio, sprout_ratio, total_defect_ratio, flags

    def _measure_color_uniformity(self, image_bgr, mask):
        """Lower std-dev in hue/saturation across the onion's own surface
        indicates a healthy, uniform skin (versus blotchy discoloration)."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        onion_pixels_mask = mask > 0
        h = hsv[:, :, 0][onion_pixels_mask]
        s = hsv[:, :, 1][onion_pixels_mask]
        if h.size == 0:
            return 0.0

        h_std = float(np.std(h))
        s_std = float(np.std(s))
        # Normalize: empirically, well-graded onions have h_std < 12, s_std < 35
        uniformity = 100 - min((h_std / 12) * 50 + (s_std / 35) * 50, 100)
        return max(uniformity, 0.0)

    # ------------------------------------------------------------------
    # Scoring & grading
    # ------------------------------------------------------------------
    def _size_band(self, diameter_mm):
        for band, (lo, hi) in self.config["size_bands_mm"].items():
            if lo <= diameter_mm < hi:
                return band
        return "unclassified"

    def _size_conformity_score(self, diameter_mm, target_band: Optional[str]):
        """
        If a target size band is specified by the procurement contract,
        score how well the onion conforms to it. Otherwise, score is
        based on being within any standard commercial band (not too
        small/damaged-looking, not oversized/split-prone).
        """
        band = self._size_band(diameter_mm)
        if target_band:
            return 100.0 if band == target_band else 40.0
        # Generic conformity: medium/large preferred commercially
        return {"small": 60.0, "medium": 95.0, "large": 90.0,
                "extra_large": 75.0, "unclassified": 30.0}.get(band, 30.0)

    def _grade_from_score(self, score):
        gt = self.config["grade_thresholds"]
        if score >= gt["A"]:
            return "A"
        elif score >= gt["B"]:
            return "B"
        elif score >= gt["C"]:
            return "C"
        return "Reject"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def grade_image(self, image_bgr: np.ndarray, target_size_band: Optional[str] = None,
                     pixels_per_mm: Optional[float] = None):
        ppm = pixels_per_mm or self.config["pixels_per_mm"]
        detections = self._segment_onions(image_bgr)

        if not detections:
            return {"onion_count": 0, "onions": [], "warning": "No onions detected in image."}

        results = []
        for idx, (contour, mask) in enumerate(detections, start=1):
            roundness, eccentricity, diam_px = self._measure_shape(contour)
            diameter_mm = diam_px / ppm
            area_mm2 = (np.pi * (diameter_mm / 2) ** 2)

            dark_ratio, sprout_ratio, defect_ratio, flags = self._measure_defects(image_bgr, mask)
            color_uniformity = self._measure_color_uniformity(image_bgr, mask)

            shape_score = max(0.0, min(roundness * 100, 100.0))
            defect_score = max(0.0, 100.0 - defect_ratio * 300)  # heavily penalize defects
            size_band = self._size_band(diameter_mm)
            size_conformity = self._size_conformity_score(diameter_mm, target_size_band)

            w = self.config["weights"]
            composite = (
                defect_score * w["defect_score"] +
                shape_score * w["shape_score"] +
                color_uniformity * w["color_uniformity_score"] +
                size_conformity * w["size_conformity_score"]
            ) / 100.0

            grade = self._grade_from_score(composite)

            if diameter_mm < 15 or diameter_mm > 150:
                flags.append("size_out_of_expected_calibration_range_verify_camera_setup")

            measurement = OnionMeasurement(
                onion_id=idx,
                equivalent_diameter_mm=round(diameter_mm, 2),
                area_mm2=round(area_mm2, 2),
                roundness=round(roundness, 3),
                eccentricity=round(eccentricity, 3),
                defect_area_ratio=round(defect_ratio, 4),
                dark_spot_ratio=round(dark_ratio, 4),
                sprout_ratio=round(sprout_ratio, 4),
                color_uniformity_score=round(color_uniformity, 2),
                size_band=size_band,
                size_conformity_score=round(size_conformity, 2),
                shape_score=round(shape_score, 2),
                defect_score=round(defect_score, 2),
                composite_score=round(composite, 2),
                grade=grade,
                flags=flags,
            )
            results.append(measurement.to_dict())

        grade_counts = {}
        for r in results:
            grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

        avg_score = round(sum(r["composite_score"] for r in results) / len(results), 2)
        lot_grade = self._grade_from_score(avg_score)

        return {
            "onion_count": len(results),
            "onions": results,
            "lot_summary": {
                "average_composite_score": avg_score,
                "lot_grade": lot_grade,
                "grade_distribution": grade_counts,
            },
            "config_version": self.config["config_version"],
        }
