"""
DAPI Image Preprocessing Module

This module provides preprocessing functions for DAPI (4',6-diamidino-2-phenylindole) 
fluorescence images to prepare them for nuclei segmentation. The preprocessing pipeline
is designed to:
- Reduce interior grain and noise while preserving nuclear boundaries
- Enhance local contrast to improve segmentation accuracy
- Standardize intensity ranges across different imaging conditions

The primary function applies a multi-step enhancement pipeline optimized for
fluorescence microscopy images of cell nuclei.
"""

import cv2
import numpy as np

def preprocess_dapi(image):
    """
    Preprocess a single‐channel DAPI image to reduce interior grain and boost edge contrast.

    Pipeline:
      1. Normalize to 0–255 and convert to uint8.
      2. Median blur (ksize=5) to remove speckle noise.
      3. Bilateral filter (d=9, σColor=75, σSpace=75) for edge‐preserving smoothing.
      4. CLAHE (clipLimit=2.0, tileGridSize=8×8) for local contrast enhancement.

    Parameters
    ----------
    image : ndarray
        2D array (any numeric type) holding the raw DAPI intensities.

    Returns
    -------
    proc : ndarray, uint8
        The preprocessed image, ready for downstream segmentation.
    """
    # 1) Normalize intensity range to full 8-bit scale (0-255)
    img = image.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img.astype(np.uint8)

    # 2) Apply median filter to remove salt-and-pepper noise and speckles
    # Kernel size 5x5 is effective for typical fluorescence noise
    med = cv2.medianBlur(img_uint8, ksize=5)

    # 3) Bilateral filter for edge-preserving smoothing
    # d=9: neighborhood diameter (~9x9 pixels)
    # sigmaColor=75: larger values mean farther colors are averaged together
    # sigmaSpace=75: larger values mean farther pixels influence each other
    bilat = cv2.bilateralFilter(med, d=9, sigmaColor=75, sigmaSpace=75)

    # 4) Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit=2.0: prevents over-amplification of noise
    # tileGridSize=(8,8): divides image into 8x8 grid for local enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc = clahe.apply(bilat)

    return proc