import numpy as np
import pandas as pd
import cv2
import cupy as cp

def analyze_nuclei(labels, fitc_image, kernel_size=51, confidenceThreshold=1.):
    """
    For each nucleus (from the DAPI segmentation) compute:
      - absolute brightness: average FITC intensity within the nucleus.
      - relative brightness: nucleus brightness minus local background brightness.
      - alive_or_dead: classification based on relative brightness and a confidence threshold.
          * Let global_std be the standard deviation of all relative brightness values.
          * If abs(rel_brightness) > (confidenceThreshold * global_std):
                status = "definitely alive" if rel_brightness > 0 else "definitely dead"
            else:
                status = "likely alive" if rel_brightness > 0 else "likely dead"

    The local background is estimated by applying a median blur (cv2.medianBlur)
    to the FITC image.

    Parameters:
      labels (ndarray): Labeled nuclei image. Can be a Cupy or NumPy array.
      fitc_image (ndarray): Corresponding FITC image.
      kernel_size (int): Kernel size for cv2.medianBlur. Must be an odd integer.
      confidenceThreshold (float): Scalar multiplier to define the threshold for definite classification.
                                   Must be >= 0.

    Returns:
      DataFrame: A pandas DataFrame with columns:
          - nucleus_id
          - absolute_brightness
          - relative_brightness
          - alive_or_dead
    """
    # Ensure labels and FITC image are on the CPU (NumPy arrays).
    if hasattr(labels, "get"):
        labels_np = cp.asnumpy(labels)
    else:
        labels_np = labels.copy()

    if hasattr(fitc_image, "get"):
        fitc_np = cp.asnumpy(fitc_image)
    else:
        fitc_np = fitc_image.copy()

    # If the FITC image is not uint8, normalize and convert it.
    if fitc_np.dtype != np.uint8:
        fitc_uint8 = cv2.normalize(fitc_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        fitc_uint8 = fitc_np

    # Compute the background image via median blur.
    # cv2.medianBlur requires the kernel size to be an odd integer.
    background_image = cv2.medianBlur(fitc_uint8, kernel_size)
    background_image = background_image.astype(np.float32)

    # Get unique nucleus labels (ignore label 0 which is background).
    nucleus_ids = np.unique(labels_np)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]

    # Initialize lists to store computed values.
    absolute_brightness_list = []
    relative_brightness_list = []
    # We'll delay classification until we've computed global_std.

    # For each nucleus, compute its absolute and relative brightness.
    for nuc_id in nucleus_ids:
        # Create a mask for the current nucleus.
        mask = (labels_np == nuc_id)

        # Compute the average brightness of the nucleus.
        abs_brightness = np.mean(fitc_np[mask].astype(np.float32))

        # Compute the local background brightness from the median-blurred image.
        bg_brightness = np.mean(background_image[mask])

        # Relative brightness is the difference.
        rel_brightness = abs_brightness - bg_brightness

        absolute_brightness_list.append(abs_brightness)
        relative_brightness_list.append(rel_brightness)

    # Compute global standard deviation of relative brightness values.
    global_std = np.std(relative_brightness_list)
    # Define our adaptive threshold.
    threshold = confidenceThreshold * global_std

    # Now, classify each nucleus using the adaptive threshold.
    alive_or_dead_list = []
    for rel_brightness in relative_brightness_list:
        if abs(rel_brightness) >= threshold:
            status = "definitely dead" if rel_brightness > 0 else "definitely alive"
        else:
            status = "likely dead" if rel_brightness > 0 else "likely alive"
        alive_or_dead_list.append(status)

    # Build a pandas DataFrame with the results.
    df = pd.DataFrame({
        "nucleus_id": nucleus_ids,
        "absolute_brightness": absolute_brightness_list,
        "relative_brightness": relative_brightness_list,
        "alive_or_dead": alive_or_dead_list
    })

    return df