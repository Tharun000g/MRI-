"""Browser upload interface for the Brain MRI enhancement project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis import calculate_metrics, metrics_to_csv
from enhancement import enhance

UPLOAD_OUTPUT_DIRECTORY = PROJECT_ROOT / "output_images" / "latest_upload"


st.set_page_config(page_title="Brain MRI Enhancer", page_icon="🧠", layout="wide")


def decode_upload(uploaded_file: Any) -> np.ndarray | None:
    """Decode an uploaded standard image format as a grayscale MRI image."""
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def image_as_png(image: np.ndarray) -> bytes:
    """Encode an OpenCV image for the download button."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not prepare the enhanced image for download.")
    return encoded.tobytes()


def show_histogram(original: np.ndarray, enhanced: np.ndarray) -> None:
    """Display an in-browser before/after histogram comparison."""
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.hist(original.ravel(), bins=256, range=(0, 256), alpha=0.55, label="Original")
    axis.hist(enhanced.ravel(), bins=256, range=(0, 256), alpha=0.55, label="Enhanced")
    axis.set_title("Intensity Histogram Comparison")
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Pixel count")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def save_latest_upload(stages: dict[str, np.ndarray], original_metrics: dict[str, float], enhanced_metrics: dict[str, float]) -> None:
    """Persist the most recent upload so it can be used in the project results."""
    UPLOAD_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    filenames = {
        "uploaded_mri_original.png": stages["original"],
        "uploaded_mri_black_background.png": stages["black_suppressed"],
        "uploaded_mri_clahe_contrast.png": stages["clahe"],
        "uploaded_mri_sharpened.png": stages["sharpened"],
        "uploaded_mri_final_enhanced.png": stages["final"],
    }
    for filename, image in filenames.items():
        cv2.imwrite(str(UPLOAD_OUTPUT_DIRECTORY / filename), image)
    (UPLOAD_OUTPUT_DIRECTORY / "uploaded_mri_quality_metrics.csv").write_text(
        metrics_to_csv(original_metrics, enhanced_metrics), encoding="utf-8"
    )


st.title("Brain MRI Contrast Enhancement")
st.caption("Near-black suppression → CLAHE local contrast enhancement → controlled sharpening")
st.warning("Educational image processing only. This app does not diagnose, detect, or segment medical conditions.")

with st.sidebar:
    st.header("Processing settings")
    black_threshold = st.slider(
        "Near-black threshold",
        min_value=0,
        max_value=50,
        value=12,
        help="Pixels at or below this value are kept at exact black throughout processing.",
    )
    st.markdown("Upload a PNG, JPG, JPEG, BMP, or TIFF brain MRI image, then select **Process image**.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is None:
    st.info("Choose an image above to begin. Nothing is processed until you upload a file and select **Process image**.")
elif st.button("Process image", type="primary"):
    image = decode_upload(uploaded_file)
    if image is None:
        st.error("The uploaded file could not be read as an image. Please use PNG, JPG, BMP, or TIFF.")
        st.stop()

    original_metrics = calculate_metrics(image)
    stages, parameters, suppressed_pixel_count = enhance(image, original_metrics, black_threshold)
    enhanced_metrics = calculate_metrics(stages["final"])
    total_pixels = stages["original"].size
    save_latest_upload(stages, original_metrics, enhanced_metrics)

    st.success(f"Processed {uploaded_file.name}. A copy of the current results was saved in output_images/latest_upload/.")
    st.write(
        f"**Settings used:** threshold {parameters.black_threshold}, CLAHE clip limit {parameters.clahe_clip_limit:.2f}, "
        f"sharpening amount {parameters.sharpen_amount:.2f}. "
        f"{suppressed_pixel_count:,} pixels ({100 * suppressed_pixel_count / total_pixels:.2f}%) were set to zero."
    )

    st.subheader("Processing stages")
    first_row = st.columns(3)
    for column, label, key in zip(first_row, ["Original", "Near-black set to zero", "CLAHE"], ["original", "black_suppressed", "clahe"]):
        with column:
            st.image(stages[key], caption=label, clamp=True, use_container_width=True)
    second_row = st.columns(2)
    for column, label, key in zip(second_row, ["Controlled sharpening", "Final enhanced MRI"], ["sharpened", "final"]):
        with column:
            st.image(stages[key], caption=label, clamp=True, use_container_width=True)

    st.subheader("Visual-quality metrics")
    metric_rows = [
        {"Metric": name, "Original": round(before, 2), "Enhanced": round(enhanced_metrics[name], 2), "Change": round(enhanced_metrics[name] - before, 2)}
        for name, before in original_metrics.items()
    ]
    st.dataframe(metric_rows, hide_index=True, use_container_width=True)
    show_histogram(stages["original"], stages["final"])

    st.subheader("Download results")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download enhanced PNG",
        data=image_as_png(stages["final"]),
        file_name="enhanced_brain_mri.png",
        mime="image/png",
    )
    download_columns[1].download_button(
        "Download metrics CSV",
        data=metrics_to_csv(original_metrics, enhanced_metrics).encode("utf-8"),
        file_name="mri_quality_metrics.csv",
        mime="text/csv",
    )
