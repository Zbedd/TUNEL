"""
Nuclei Labeling and Segmentation

This module provides functions for segmenting nuclei in DAPI-stained images
using either traditional Otsu thresholding or modern YOLO deep learning models.
"""
import numpy as np
import cv2
import pyclesperanto_prototype as cle
import scipy.ndimage as ndi
from pathlib import Path
from . import DEFAULTS

# Optional cupy import for GPU acceleration when available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, disk


def segmentation_pipeline_otsu(input_image, *, splitting=True):
    """
    Traditional nuclei segmentation using Otsu thresholding.
    
    Process: DAPI -> Gaussian blur -> Otsu threshold -> hole filling -> 
             morphological closing -> optional watershed splitting -> 
             GPU-accelerated Voronoi-Otsu labeling
    
    Args:
        input_image: DAPI image as numpy array
        splitting: Whether to apply watershed splitting to separate touching nuclei
        
    Returns:
        tuple: (labels, binary_mask) where labels are instance segmentations
    """
    # Smooth the image to reduce noise before thresholding
    blur = cv2.GaussianBlur(input_image, (5, 5), 2)
    
    # Apply Otsu's method to find optimal threshold
    binary = blur > threshold_otsu(blur)
    
    # Clean up the binary mask
    binary = ndi.binary_fill_holes(binary)
    binary = binary_closing(binary, footprint=disk(3))

    if splitting:
        # Use watershed to separate touching nuclei
        dist = ndi.distance_transform_edt(binary)
        local_max = ndi.maximum_filter(dist, size=5) == dist
        markers, _ = ndi.label(local_max)
        split = watershed(-dist, markers, mask=binary)
        binary = split > 0

    # Generate instance labels using GPU-accelerated Voronoi-Otsu
    labels = cle.voronoi_otsu_labeling(binary, spot_sigma=7, outline_sigma=0.1)
    return labels, binary


# Import YOLO functionality from external imageProcessingUtils package
try:
    from imageProcessingUtils.yolo import segmentation_pipeline_yolo as external_yolo_pipeline
    print("YOLO module loaded successfully from imageProcessingUtils")
except ImportError as e:
    print(f"Warning: could not import YOLO module from imageProcessingUtils: {e}")
    print("         YOLO-based segmentation will be unavailable.")
    external_yolo_pipeline = None


def segmentation_pipeline_yolo(input_image, *, conf_thres=0.01):
    """
    Deep learning-based nuclei segmentation using YOLO model.
    
    Args:
        input_image: DAPI image as numpy array
        conf_thres: Confidence threshold for detections (lower = more sensitive)
        
    Returns:
        tuple: (labels, binary_mask) from YOLO segmentation
        
    Raises:
        RuntimeError: If YOLO model is not available
    """
    if external_yolo_pipeline is None:
        raise RuntimeError("YOLO model not loaded; cannot run YOLO segmentation.")
    
    return external_yolo_pipeline(input_image, conf_thres=conf_thres)


def label_nuclei(
    dapi_image: np.ndarray,
    *,
    method: str = "otsu",
    iterate: bool = False,
    splitting: bool = True,
    remove_small_outliers: bool = False,
    remove_large_outliers: bool = False,
    min_label_area: int = 250,
    # Parameters for iterative enhancement
    initial_clipLimit: float = 2.0,
    max_clip_iterations: int = 10,
    max_baseline_size: int = 80_000,
    return_binary: bool = False,
    verbose: bool = False,
    apply_masking: bool = False,
    mask_folder: Path = None,
    name: str = None
):
    """
    Main nuclei labeling function with two operating modes.
    
    This function can operate in basic mode (single segmentation pass) or
    iterative mode (baseline + CLAHE enhancement loop for challenging images).
    
    Args:
        dapi_image: Input DAPI image as numpy array
        method: Segmentation method - 'otsu' for traditional, 'yolo' for deep learning
        iterate: If False, single pass; if True, iterative enhancement
        splitting: Whether to separate touching nuclei (Otsu only)
        remove_small_outliers: Filter out very small detections
        remove_large_outliers: Filter out very large detections
        min_label_area: Minimum area for valid nuclei (pixels)
        initial_clipLimit: Starting CLAHE clip limit for iterative mode
        max_clip_iterations: Maximum CLAHE iterations
        max_baseline_size: Target max nucleus size for iteration stopping
        return_binary: Whether to return binary mask along with labels
        verbose: Print iteration details
        apply_masking: Apply tissue mask if available
        mask_folder: Path to directory containing mask files
        name: Original image filename for mask matching
        
    Returns:
        labels: Instance segmentation labels
        binary: Binary mask (if return_binary=True)
    """

    # Choose segmentation method and parameters
    if method == "otsu":
        seg_fn, seg_kw = segmentation_pipeline_otsu, {"splitting": splitting}
    elif method == "yolo":
        seg_fn, seg_kw = segmentation_pipeline_yolo, {"conf_thres": 0.01}
    else:
        raise ValueError("method must be 'otsu' or 'yolo'")

    # Configure GPU acceleration if available
    cle.select_device("cupy")
    img = np.asarray(dapi_image)

    # Basic mode: single segmentation pass
    if not iterate:
        labels, binary = seg_fn(img, **seg_kw)

    # Iterative mode: enhance difficult images with CLAHE
    else:
        # Step 1: Get baseline segmentation without splitting
        base_kw = seg_kw.copy()
        if method == "otsu":
            base_kw["splitting"] = False  # Defer splitting until final pass
            
        base_lbl, base_bin = seg_fn(img, **base_kw)
        baseline_max = np.bincount(base_bin.ravel())[1:].max(initial=0)
        if verbose:
            print(f"[baseline] largest connected component = {baseline_max}")

        # Step 2: CLAHE enhancement loop to break up large regions
        clip = initial_clipLimit
        best = (baseline_max, img)  # Track best (smallest_max, image) pair

        for i in range(max_clip_iterations):
            # Apply Contrast Limited Adaptive Histogram Equalization
            clahe_img = cv2.createCLAHE(clip, (8, 8)).apply(img.astype("uint8"))
            _, bin_tmp = seg_fn(clahe_img, **base_kw)
            cur_max = np.bincount(bin_tmp.ravel())[1:].max(initial=0)

            if verbose:
                print(f"  iter {i:2d}: clip={clip:.1f}  max_component={cur_max}")

            # Keep track of the best enhancement so far
            if cur_max < best[0]:
                best = (cur_max, clahe_img)
                
            # Stop if we've achieved our target
            if cur_max <= max_baseline_size:
                img = clahe_img
                break

            # Reduce clip limit for next iteration
            clip = max(round(clip - 0.2, 2), 0.2)
        else:
            # Loop completed without reaching target - use best result
            img = best[1] if best[0] < baseline_max else img

        # Step 3: Final segmentation with original parameters
        labels, binary = seg_fn(img, **seg_kw)

    # ========================================================================== 
    #  COMMON POST-PROCESSING: Statistical outlier removal and min-area filter
    # ========================================================================== 
    # Convert labels to CPU numpy array for statistical analysis
    arr = labels.get() if hasattr(labels, "get") else labels
    
    # Calculate area of each segmented nucleus (excluding background label 0)
    areas = np.bincount(arr.ravel())[1:]
    
    if areas.size:
        # Calculate interquartile range (IQR) for outlier detection
        Q1, Q3 = np.percentile(areas, [25, 75])
        IQR = Q3 - Q1
        
        # Define outlier bounds using modified IQR method (0.5x instead of 1.5x for tighter bounds)
        low, high = Q1 - 0.5 * IQR, Q3 + 0.5 * IQR
        
        # Create boolean mask to track which labels to keep
        keep = np.ones_like(arr, bool)

        # Filter labels based on size criteria
        for lid, a in enumerate(areas, 1):
            should_remove = (
                (remove_small_outliers and a < low) or      # Remove statistical small outliers
                (remove_large_outliers and a > high) or     # Remove statistical large outliers  
                (a < min_label_area)                         # Remove below minimum area threshold
            )
            if should_remove:
                keep[arr == lid] = False

        # Remove filtered labels by setting them to background (0)
        arr[~keep] = 0
        
        # Relabel sequentially to close gaps in label numbering
        # Use GPU if available and original was GPU array, otherwise use CPU
        if HAS_CUPY and hasattr(labels, "get"):
            labels = cle.relabel_sequential(cp.asarray(arr))
        else:
            labels = cle.relabel_sequential(arr)

    # --------------------------------------------------------------------------
    # Mask-based spatial filtering: Remove nuclei outside defined regions
    # --------------------------------------------------------------------------
    if apply_masking:
        if mask_folder is None:
            if verbose:
                print("⚠️  Warning: apply_masking=True but mask_folder is None; skipping masking")
        elif name is None:
            raise ValueError("mask_name must be provided when apply_masking=True")
        else:
            # Derive filename stem (remove extension if present) for mask lookup
            stem = Path(name).stem
            mask_file = mask_folder / f"{stem}_mask.tif"
            
            # Attempt to load corresponding spatial mask
            try:
                mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            except Exception:
                mask_img = None
                
            if mask_img is None:
                if verbose:
                    print(f"⚠️  Warning: mask not found or unreadable at {mask_file}; skipping apply_masking")
            else:
                # Prepare mask as binary region of interest (ROI)
                mask_img = np.squeeze(mask_img)
                roi = mask_img > 0
                if roi.ndim == 3:
                    roi = roi[..., 0]  # Take first channel if RGB
                
                # Convert labels to CPU array for mask processing
                arr = labels.get() if hasattr(labels, "get") else np.array(labels)
                keep = np.ones_like(arr, bool)
                
                # Check each nucleus for spatial overlap with ROI
                for lid in np.unique(arr):
                    if lid == 0:  # Skip background
                        continue
                    
                    # Calculate overlap percentage with ROI
                    nucleus = (arr == lid)
                    total_pixels = nucleus.sum()
                    inside_pixels = np.logical_and(nucleus, roi).sum()
                    outside_fraction = (total_pixels - inside_pixels) / total_pixels
                    
                    # Remove nucleus if more than 10% is outside the ROI
                    if outside_fraction > 0.10:
                        keep[nucleus] = False
                
                # Apply mask filtering and relabel
                arr[~keep] = 0
                if HAS_CUPY and hasattr(labels, "get"):
                    labels = cle.relabel_sequential(cp.asarray(arr))
                else:
                    labels = cle.relabel_sequential(arr)
                                    
    # Calculate final statistics for all labeled nuclei
    stats = cle.statistics_of_labelled_pixels(dapi_image, labels)

    # Return results based on requested output format
    if return_binary:
        return labels, stats, binary
    return labels, stats

# Note: Alternative basic labeling function was removed in favor of the more 
# comprehensive label_nuclei() function above which includes all functionality
# plus iterative enhancement capabilities.