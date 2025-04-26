#Imports
import numpy as np
import cv2
import pyclesperanto_prototype as cle
from ultralytics import YOLO
import scipy.ndimage as ndi
import cupy as cp
from pathlib import Path


from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, disk

    
# NEXT STEPS
#     • Integrate otsu and YOLO segmentation methods, potentially generating otsu through iteration. New function: label_nuclei_integrated


# --------------------------------------------------------------------
# segmentation using otsu
# --------------------------------------------------------------------
def segmentation_pipeline_otsu(input_image, *, splitting=True):
    """
    Classic DAPI→Otsu threshold → holes/closing → optional watershed split →
    GPU Voronoi-Otsu labelling.
    
    Will ignore splitting, as splitting is handled by yolo directly
    """
    blur   = cv2.GaussianBlur(input_image, (5,5), 2)
    binary = blur > threshold_otsu(blur)
    binary = ndi.binary_fill_holes(binary)
    binary = binary_closing(binary, footprint=disk(3))

    if splitting:
        dist      = ndi.distance_transform_edt(binary)
        local_max = ndi.maximum_filter(dist, size=5) == dist
        markers, _= ndi.label(local_max)
        split     = watershed(-dist, markers, mask=binary)
        binary    = split > 0

    labels = cle.voronoi_otsu_labeling(binary, spot_sigma=7, outline_sigma=0.1)
    return labels, binary

# --------------------------------------------------------------------
# segmentation using YOLO (first build model through yolo_model_training.py)
# --------------------------------------------------------------------
YOLO_MODEL = None
path_base = Path(r"C:/VScode/TUNEL/runs/segment")   # root that holds train, train2, …

# find every best.pt, keep the one whose parent run folder is newest
YOLO_PATH = max(
    path_base.glob("train*/weights/best.pt"),
    key=lambda p: p.stat().st_mtime,             # most-recent modification time
)

try:
    YOLO_MODEL = YOLO(YOLO_PATH)
    YOLO_MODEL.fuse()
    print(f"✅ YOLO model loaded from '{YOLO_PATH}'")
except Exception as e:
    print(f"⚠️  Warning: could not load YOLO model at '{YOLO_PATH}': {e}\n"
          "         YOLO-based segmentation will be unavailable.")

def segmentation_pipeline_yolo(input_image, *, conf_thres=0.01):
    """YOLO-v8 retina_masks segmentation → union mask + instance labels."""
    if YOLO_MODEL is None:
        raise RuntimeError("YOLO model not loaded; cannot run YOLO segmentation.")

    # prepare 3-ch uint8
    img8 = np.clip(input_image, 0, 255).astype(np.uint8)
    if img8.ndim == 2:
        img8 = np.stack([img8]*3, axis=-1)

    # inference
    res   = YOLO_MODEL(img8, imgsz=768, mask_ratio = 1, conf=conf_thres, retina_masks=True, verbose=False)[0]
    masks = res.masks.data  # Tensor (N, H, W)

    h, w = input_image.shape
    if masks is None or masks.shape[0] == 0:
        binary = np.zeros((h, w), dtype=bool)
        labels = np.zeros((h, w), dtype=int)
    else:
        mb     = masks.bool().cpu()
        h, w = input_image.shape
        labels = np.zeros((h,w),int)
        for i in range(mb.shape[0]):
            labels[ mb[i].cpu().numpy() ] = i+1
        binary = labels>0

    return labels, binary

# --------------------------------------------------------------------
# core labeling function
# --------------------------------------------------------------------
def label_nuclei(
    dapi_image: np.ndarray,
    *,
    method: str = "otsu",            # 'otsu' | 'yolo'
    iterate: bool = False,           # False → basic ; True → iterative
    splitting: bool = True,
    remove_small_outliers: bool = False,
    remove_large_outliers: bool = False,
    min_label_area: int = 250,
    # iterative-mode knobs
    initial_clipLimit: float = 2.0,
    max_clip_iterations: int = 10,
    max_baseline_size: int = 80_000,
    return_binary: bool = False,
    verbose: bool = False,
):
    """
    One function, two behaviours:
    ----------------------------------
    • iterate=False  → single segmentation pass (old label_nuclei_basic)
    • iterate=True   → baseline + CLAHE loop (old label_nuclei)
    """

    # ---------- choose backend ------------------------------------------------
    if method == "otsu":
        seg_fn, seg_kw = segmentation_pipeline_otsu, {"splitting": splitting}
    elif method == "yolo":
        seg_fn, seg_kw = segmentation_pipeline_yolo, {"conf_thres": 0.01}
    else:
        raise ValueError("method must be 'otsu' or 'yolo'")

    cle.select_device("cupy")
    img = np.asarray(dapi_image)

    # ========================================================================== 
    #  BASIC PATH  (iterate == False)
    # ========================================================================== 
    if not iterate:
        labels, binary = seg_fn(img, **seg_kw)

    # ========================================================================== 
    #  ITERATIVE PATH  (iterate == True)
    # ========================================================================== 
    else:
        # --- STEP-1 baseline (no splitting) -----------------------------------
        # For baseline, override splitting to False even if method=='otsu'
        base_kw = seg_kw.copy()
        if method == "otsu":
            base_kw["splitting"] = False
        base_lbl, base_bin = seg_fn(img, **base_kw)
        baseline_max = np.bincount(base_bin.ravel())[1:].max(initial=0)
        if verbose:
            print(f"[baseline] largest cc = {baseline_max}")

        # --- CLAHE loop -------------------------------------------------------
        clip  = initial_clipLimit
        best  = (baseline_max, img)          # (best_max, best_image)

        for i in range(max_clip_iterations):
            clahe_img = cv2.createCLAHE(clip, (8, 8)).apply(img.astype("uint8"))
            _, bin_tmp = seg_fn(clahe_img, **base_kw)
            cur_max    = np.bincount(bin_tmp.ravel())[1:].max(initial=0)

            if verbose:
                print(f"  iter {i:2d}: clip={clip:.1f}  max={cur_max}")

            if cur_max < best[0]:
                best = (cur_max, clahe_img)
            if cur_max <= max_baseline_size:
                img = clahe_img           # accept
                break

            clip = max(round(clip - 0.2, 2), 0.2)
        else:                               # loop exhausted
            img = best[1] if best[0] < baseline_max else img

        # --- STEP-2 final segmentation (with splitting flag) ------------------
        labels, binary = seg_fn(img, **seg_kw)

    # ========================================================================== 
    #  COMMON POST-PROCESSING  (outlier + min-area filter) 
    # ========================================================================== 
    arr   = labels.get() if hasattr(labels, "get") else labels
    areas = np.bincount(arr.ravel())[1:]
    if areas.size:
        Q1, Q3 = np.percentile(areas, [25, 75]); IQR = Q3 - Q1
        low, high = Q1 - 0.5 * IQR, Q3 + 0.5 * IQR
        keep = np.ones_like(arr, bool)

        for lid, a in enumerate(areas, 1):
            if (remove_small_outliers and a < low) or \
               (remove_large_outliers and a > high) or \
               (a < min_label_area):
                keep[arr == lid] = False

        arr[~keep] = 0
        labels = cle.relabel_sequential(cp.asarray(arr) if hasattr(labels, "get") else arr)

    stats = cle.statistics_of_labelled_pixels(dapi_image, labels)

    if return_binary:
        return labels, stats, binary
    return labels, stats

# def label_nuclei_basic(dapi_image, *, 
#                        method = 'otsu',     
#                        splitting=True,
#                        remove_small_outliers=False,
#                        remove_large_outliers=False,
#                        return_binary=False,
#                        min_label_area=250,
#                        verbose=False):
#     """
#     dapi_image : 2D ndarray

#     method : str
#       "otsu" to use classic Otsu thresholding
#       "yolo" to use the YOLO-v8 retina_masks segmentation

#     The rest of the parameters control the CLAHE contrast-loop,
#     optional splitting, outlier filtering, and min-area filtering.

#     Returns:
#       final_labels : labeled ndarray
#       label_stats  : dict from cle.statistics_of_labelled_pixels
#       (optional final_binary mask)
#     """
#     # pick the pipeline
#     if method == "otsu":
#         seg_fn = segmentation_pipeline_otsu
#         seg_kwargs = {"splitting": splitting}
#     elif method == "yolo":
#         seg_fn = segmentation_pipeline_yolo
#         seg_kwargs = {"splitting": splitting, "conf_thres": 0.1}
#     else:
#         raise ValueError(f"Unknown method {method!r}; choose 'otsu' or 'yolo'")

#     cle.select_device("cupy")
#     img = np.copy(dapi_image)

#     # Generate binary mask
#     labels, binary = seg_fn(img, **seg_kwargs)
#     cc, _        = ndi.label(binary)
#     sizes        = np.bincount(cc.ravel())[1:]
#     max_area = sizes.max() if sizes.size else 0
#     if verbose:
#         print(f"Largest label = {max_area}")

#     # outlier + min-area filters
#     arr = labels.get() if hasattr(labels, "get") else labels
#     areas = np.bincount(arr.ravel())[1:]
#     if areas.size:
#         Q1, Q3 = np.percentile(areas, [25,75]); IQR = Q3 - Q1
#         low, high = Q1 - 0.5*IQR, Q3 + 0.5*IQR
#         filt = arr.copy()
#         for lid, a in enumerate(areas, 1):
#             if (remove_small_outliers and a < low) or \
#                (remove_large_outliers and a > high) or \
#                (a < min_label_area):
#                 filt[arr == lid] = 0
#         labels = cle.relabel_sequential(
#             cp.asarray(filt) if hasattr(labels, "get") else filt
#         )

#     stats = cle.statistics_of_labelled_pixels(dapi_image, labels)

#     if return_binary:
#         return labels, stats, binary
#     return binary, stats