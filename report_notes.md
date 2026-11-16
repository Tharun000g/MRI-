# Report Notes: Brain MRI Contrast Enhancement and Quality Analysis

## Abstract

This project implements a non-machine-learning method for improving the visual presentation of a 2D brain MRI. The proposed pipeline first suppresses near-black background pixels by setting them to zero, then applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve local contrast, followed by controlled unsharp-mask sharpening. Unlike the earlier approach, no denoising, contour extraction, or edge-map output is used, preventing smoothing from reducing visible structures. The project generates stage-by-stage images, a comparison panel, intensity histograms, and a before/after quality-metric table. The method is intended for educational digital image processing and not clinical diagnosis.

## Problem statement

Brain MRI images may have low local contrast and unwanted faint near-black background values. Simple smoothing can reduce noise but may also blur fine anatomical structures. This project therefore focuses on contrast enhancement and sharpening without denoising.

## Objectives

1. Set near-black background pixels to exact black for clearer display contrast.
2. Improve local tissue contrast using CLAHE.
3. Emphasize visible boundaries using controlled sharpening.
4. Compare the original and final images using visual-quality metrics.

## Methodology

1. Load and normalize a grayscale brain MRI to the 0-255 intensity range.
2. Create a background mask for pixels at or below the selected black threshold.
3. Set masked pixels to zero.
4. Apply CLAHE to enhance local contrast.
5. Reapply the background mask to preserve true black pixels.
6. Apply unsharp masking for controlled sharpening.
7. Reapply the background mask once more.
8. Compare the original and final images using contrast, robust dynamic range, entropy, edge strength, and background-pixel percentage.

## Original contribution

The key design contribution is background-preserving enhancement. The program uses one near-black mask throughout the pipeline. As a result, CLAHE and sharpening enhance the brain region while the selected dark background remains exactly zero. The threshold can be changed from the command line for different images.

## Results discussion

Use `full_comparison_panel.png` to explain each processing stage, and use `quality_metrics.csv` to report numerical changes. Increased global contrast or robust dynamic range can support improved tonal separation. Increased edge strength can support a sharper appearance. Metrics should be discussed together with the visual comparison because no one number proves medical usefulness.

## Conclusion

The project demonstrates a focused MRI image-enhancement pipeline using near-black intensity suppression, CLAHE, and controlled sharpening. It preserves a clean black background while enhancing local visual contrast and edge definition, without relying on ML or smoothing filters.

## Limitation and ethics statement

This is an educational visual-enhancement project. It does not identify, segment, classify, or diagnose tumors or other medical conditions, and it must not be used for clinical decisions.
