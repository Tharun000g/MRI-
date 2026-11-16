"""Classical contrast enhancement for 2D brain MRI images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from analysis import normalize_to_uint8


@dataclass(frozen=True)
class EnhancementParameters:
    """Settings used by the enhancement pipeline."""

    black_threshold: int
    clahe_clip_limit: float
    sharpen_amount: float


def choose_parameters(metrics: dict[str, float], black_threshold: int) -> EnhancementParameters:
    """Use image contrast to select a conservative CLAHE strength."""
    dynamic_range = metrics["Robust dynamic range (P95-P5)"]
    clip_limit = 3.2 if dynamic_range < 120 else 2.2
    return EnhancementParameters(black_threshold=black_threshold, clahe_clip_limit=clip_limit, sharpen_amount=0.85)


def suppress_near_black(image: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Set near-black background pixels to exactly zero and return their mask."""
    image = normalize_to_uint8(image)
    background_mask = image <= threshold
    suppressed = image.copy()
    suppressed[background_mask] = 0
    return suppressed, background_mask


def apply_clahe(image: np.ndarray, clip_limit: float) -> np.ndarray:
    """Improve local tissue contrast with contrast-limited equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(normalize_to_uint8(image))


def unsharp_mask(image: np.ndarray, amount: float, radius: float = 1.0) -> np.ndarray:
    """Apply controlled sharpening without any denoising stage."""
    image = normalize_to_uint8(image)
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def enhance(image: np.ndarray, metrics: dict[str, float], black_threshold: int) -> tuple[dict[str, np.ndarray], EnhancementParameters, int]:
    """Run black-level suppression, CLAHE, and sharpening."""
    params = choose_parameters(metrics, black_threshold)
    original = normalize_to_uint8(image)
    black_suppressed, background_mask = suppress_near_black(original, params.black_threshold)
    clahe = apply_clahe(black_suppressed, params.clahe_clip_limit)
    clahe[background_mask] = 0
    sharpened = unsharp_mask(clahe, params.sharpen_amount)
    sharpened[background_mask] = 0
    return {"original": original, "black_suppressed": black_suppressed, "clahe": clahe, "sharpened": sharpened, "final": sharpened}, params, int(np.count_nonzero(background_mask))
