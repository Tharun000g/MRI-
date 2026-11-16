"""Command-line entry point for brain MRI contrast enhancement."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import calculate_metrics, save_histogram_comparison, save_metric_chart, save_metrics_csv
from enhancement import enhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "input_images"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output_images"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_input_image() -> Path:
    """Prefer the included public-domain brain MRI; otherwise use the first image."""
    preferred = INPUT_DIRECTORY / "brain_mri_t1.png"
    if preferred.is_file():
        return preferred
    candidates = sorted(path for path in INPUT_DIRECTORY.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not candidates:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(f"No supported image found in {INPUT_DIRECTORY}. Add one of: {extensions}")
    return candidates[0]


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")


def create_comparison_panel(stages: dict[str, np.ndarray], output_path: Path) -> None:
    """Save the five-stage visual comparison used in the project demo."""
    panels = [
        ("Original brain MRI", stages["original"]),
        ("Near-black pixels set to zero", stages["black_suppressed"]),
        ("CLAHE local contrast", stages["clahe"]),
        ("Controlled sharpening", stages["sharpened"]),
        ("Final enhanced MRI", stages["final"]),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(20, 5))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle("Brain MRI Enhancement: Background Suppression, CLAHE, and Sharpening", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance a 2D brain MRI using classical image processing.")
    parser.add_argument("--input", type=Path, help="Path to an MRI image. Defaults to input_images/brain_mri_t1.png.")
    parser.add_argument("--black-threshold", type=int, default=12, choices=range(0, 51), metavar="0-50", help="Set pixels at or below this intensity to zero (default: 12).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve() if args.input else find_input_image()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {input_path}")

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read {input_path}. Use PNG, JPG, BMP, or TIFF.")

    original_metrics = calculate_metrics(image)
    stages, params, suppressed_pixel_count = enhance(image, original_metrics, args.black_threshold)
    enhanced_metrics = calculate_metrics(stages["final"])

    for filename, result in {
        "brain_mri_original.png": stages["original"],
        "brain_mri_black_background.png": stages["black_suppressed"],
        "brain_mri_clahe_contrast.png": stages["clahe"],
        "brain_mri_sharpened.png": stages["sharpened"],
        "brain_mri_final_enhanced.png": stages["final"],
    }.items():
        save_image(OUTPUT_DIRECTORY / filename, result)

    save_histogram_comparison(stages["original"], stages["final"], OUTPUT_DIRECTORY / "brain_mri_histogram_comparison.png")
    save_metric_chart(original_metrics, enhanced_metrics, OUTPUT_DIRECTORY / "brain_mri_quality_metrics.png")
    save_metrics_csv(original_metrics, enhanced_metrics, OUTPUT_DIRECTORY / "brain_mri_quality_metrics.csv")
    create_comparison_panel(stages, OUTPUT_DIRECTORY / "brain_mri_processing_comparison.png")

    print(f"Processed: {input_path.name}")
    print("Pipeline: near-black suppression -> CLAHE -> controlled sharpening")
    print("Settings:")
    print(f"  Black threshold: {params.black_threshold}")
    print(f"  Pixels set to zero: {suppressed_pixel_count} ({100 * suppressed_pixel_count / stages['original'].size:.2f}%)")
    print(f"  CLAHE clip limit: {params.clahe_clip_limit:.2f}")
    print(f"  Sharpening amount: {params.sharpen_amount:.2f}")
    print("\nMetric comparison:")
    for metric, before in original_metrics.items():
        print(f"  {metric}: {before:.2f} -> {enhanced_metrics[metric]:.2f}")
    print(f"\nResults saved in: {OUTPUT_DIRECTORY}")
    print("Educational image enhancement only; this is not a diagnostic tool.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
