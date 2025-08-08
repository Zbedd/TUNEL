"""
Nuclei Analysis and Classification Module

This module provides functions for analyzing segmented nuclei to determine cell viability
based on fluorescence intensity patterns. The primary application is TUNEL staining
analysis, where FITC fluorescence indicates DNA fragmentation (cell death).

Key functionality:
- Compute absolute and relative brightness of each nucleus
- Classify nuclei as alive/dead with confidence levels
- Background subtraction using local median filtering
- Statistical thresholding for classification confidence

The analysis pipeline compares nucleus brightness to local background and uses
global statistics to establish confidence thresholds for cell death classification.
"""

import numpy as np
import pandas as pd
import cv2

# Optional cupy import for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def analyze_nuclei(labels, fitc_image, kernel_size=51, confidenceThreshold=1.):
    """
    Analyze segmented nuclei to classify cell viability based on FITC fluorescence.
    
    This function performs TUNEL staining analysis by comparing nucleus fluorescence
    intensity to local background. High FITC signal indicates DNA fragmentation 
    (cell death), while low signal suggests cell viability.

    For each nucleus, the analysis computes:
      - Absolute brightness: Average FITC intensity within the nucleus
      - Relative brightness: Nucleus brightness minus local background brightness  
      - Classification: Alive/dead determination with confidence levels

    Classification algorithm:
      1. Calculate global standard deviation of all relative brightness values
      2. Define threshold = confidenceThreshold × global_std
      3. Classify based on threshold:
         - If |rel_brightness| > threshold: "definitely alive/dead"
         - If |rel_brightness| ≤ threshold: "likely alive/dead"
      4. Direction: positive rel_brightness = dead, negative = alive

    Parameters
    ----------
    labels : ndarray
        Labeled nuclei image from segmentation. Can be CuPy or NumPy array.
        Background should be labeled as 0, nuclei as positive integers.
    fitc_image : ndarray  
        Corresponding FITC fluorescence image. Should match labels dimensions.
    kernel_size : int, default=51
        Kernel size for median blur background estimation. Must be odd integer.
        Larger values create smoother background estimation.
    confidenceThreshold : float, default=1.0
        Multiplier for statistical threshold. Higher values require stronger
        signal for "definite" classification. Must be >= 0.

    Returns
    -------
    DataFrame
        Pandas DataFrame with analysis results containing columns:
        - nucleus_id: Unique identifier for each nucleus
        - absolute_brightness: Mean FITC intensity within nucleus
        - relative_brightness: Nucleus intensity minus local background
        - alive_or_dead: Classification string ("definitely/likely alive/dead")
    """
    # Convert GPU arrays to CPU numpy arrays for OpenCV compatibility
    if HAS_CUPY and hasattr(labels, "get"):
        labels_np = cp.asnumpy(labels)
    else:
        labels_np = labels.copy()

    if HAS_CUPY and hasattr(fitc_image, "get"):
        fitc_np = cp.asnumpy(fitc_image)
    else:
        fitc_np = fitc_image.copy()

    # Normalize FITC image to uint8 format if necessary for OpenCV processing
    if fitc_np.dtype != np.uint8:
        fitc_uint8 = cv2.normalize(fitc_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        fitc_uint8 = fitc_np

    # Estimate local background using median filter
    # Median blur effectively removes small bright objects while preserving larger structures
    background_image = cv2.medianBlur(fitc_uint8, kernel_size)
    background_image = background_image.astype(np.float32)

    # Extract unique nucleus labels (exclude background label 0)
    nucleus_ids = np.unique(labels_np)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]

    # Initialize storage for brightness measurements
    absolute_brightness_list = []
    relative_brightness_list = []

    # Compute brightness metrics for each nucleus
    for nuc_id in nucleus_ids:
        # Create binary mask for current nucleus
        mask = (labels_np == nuc_id)

        # Calculate absolute brightness: mean FITC intensity within nucleus
        abs_brightness = np.mean(fitc_np[mask].astype(np.float32))

        # Calculate local background: mean intensity from median-filtered image
        bg_brightness = np.mean(background_image[mask])

        # Relative brightness: difference between nucleus and local background
        rel_brightness = abs_brightness - bg_brightness

        absolute_brightness_list.append(abs_brightness)
        relative_brightness_list.append(rel_brightness)

    # Calculate global statistics for classification thresholding
    global_std = np.std(relative_brightness_list)
    threshold = confidenceThreshold * global_std

    # Classify each nucleus based on relative brightness and confidence threshold
    alive_or_dead_list = []
    for rel_brightness in relative_brightness_list:
        # Strong signal above threshold = definite classification
        if abs(rel_brightness) >= threshold:
            # Positive rel_brightness = brighter than background = dead cell
            # Negative rel_brightness = dimmer than background = alive cell  
            status = "definitely dead" if rel_brightness > 0 else "definitely alive"
        else:
            # Weak signal below threshold = likely classification
            status = "likely dead" if rel_brightness > 0 else "likely alive"
        alive_or_dead_list.append(status)

    # Compile results into structured DataFrame
    df = pd.DataFrame({
        "nucleus_id": nucleus_ids,
        "absolute_brightness": absolute_brightness_list,
        "relative_brightness": relative_brightness_list,
        "alive_or_dead": alive_or_dead_list
    })

    return df