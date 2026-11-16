"""Non-diagnostic image measurements and result visualizations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import cv2

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import shannon_entropy


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Return a single-channel image scaled to the 0-255 range."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def calculate_metrics(image: np.ndarray) -> dict[str, float]:
    """Calculate visual-quality descriptors; they are not clinical measures."""
    image = normalize_to_uint8(image)
    p5, p95 = np.percentile(image, [5, 95])
    return {
        "Mean intensity": float(np.mean(image)),
        "Global contrast (standard deviation)": float(np.std(image)),
        "Robust dynamic range (P95-P5)": float(p95 - p5),
        "Entropy": float(shannon_entropy(image)),
        "Edge strength (Laplacian variance)": float(cv2.Laplacian(image, cv2.CV_64F).var()),
        "Zero-value pixels (%)": float(100 * np.count_nonzero(image == 0) / image.size),
    }


def save_histogram_comparison(original: np.ndarray, enhanced: np.ndarray, output_path: Path) -> None:
    """Save original versus final intensity distributions."""
    plt.figure(figsize=(10, 5))
    plt.hist(normalize_to_uint8(original).ravel(), bins=256, range=(0, 256), alpha=0.55, label="Original")
    plt.hist(normalize_to_uint8(enhanced).ravel(), bins=256, range=(0, 256), alpha=0.55, label="Final enhanced")
    plt.title("Brain MRI Intensity Histogram Comparison")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metric_chart(original: Mapping[str, float], enhanced: Mapping[str, float], output_path: Path) -> None:
    """Save a separate before/after bar chart for each metric."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, name in zip(axes.flat, original):
        values = [original[name], enhanced[name]]
        bars = axis.bar(["Original", "Enhanced"], values, color=["#607d8b", "#1565c0"])
        axis.set_title(name, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Visual-Quality Metric Comparison (Non-diagnostic)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def metrics_to_csv(original: Mapping[str, float], enhanced: Mapping[str, float]) -> str:
    """Create a report-ready before/after metric table as CSV text."""
    notes = {
        "Mean intensity": "Brightness distribution",
        "Global contrast (standard deviation)": "Higher can indicate greater overall tonal separation",
        "Robust dynamic range (P95-P5)": "Higher indicates a wider useful intensity span",
        "Entropy": "Describes intensity-distribution complexity",
        "Edge strength (Laplacian variance)": "Higher can indicate stronger visible edges",
        "Zero-value pixels (%)": "Background suppression; not a standalone quality score",
    }
    lines = ["Metric,Original,Enhanced,Change,Interpretation"]
    for name, before in original.items():
        after = enhanced[name]
        lines.append(f'"{name}",{before:.4f},{after:.4f},{after - before:.4f},"{notes[name]}"')
    return "\n".join(lines) + "\n"


def save_metrics_csv(original: Mapping[str, float], enhanced: Mapping[str, float], output_path: Path) -> None:
    """Save a report-ready before/after metric table."""
    output_path.write_text(metrics_to_csv(original, enhanced), encoding="utf-8")
