#Imports
import numpy as np
import cv2
import pyclesperanto_prototype as cle
import torch
from ultralytics import YOLO
import scipy.ndimage as ndi
import cupy as cp


from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, disk


# --------------------------------------------------------------------
# segmentation using otsu
# --------------------------------------------------------------------
def segmentation_pipeline_otsu(input_image, *, splitting=True):
    """
    Classic DAPI→Otsu threshold → holes/closing → optional watershed split →
    GPU Voronoi-Otsu labelling.
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
YOLO_PATH  = "C:/VScode/TUNEL/runs/segment/train/weights/best.pt"  # adjust as needed
try:
    YOLO_MODEL = YOLO(YOLO_PATH)
    YOLO_MODEL.fuse()
except Exception as e:
    print(f"⚠️  Warning: could not load YOLO model at '{YOLO_PATH}': {e}\n"
          "         YOLO-based segmentation will be unavailable.")

def segmentation_pipeline_yolo(input_image, *, splitting=True, conf_thres=0.1):
    """YOLO-v8 retina_masks segmentation → union mask + instance labels."""
    if YOLO_MODEL is None:
        raise RuntimeError("YOLO model not loaded; cannot run YOLO segmentation.")

    # prepare 3-ch uint8
    img8 = np.clip(input_image, 0, 255).astype(np.uint8)
    if img8.ndim == 2:
        img8 = np.stack([img8]*3, axis=-1)

    # inference
    res   = YOLO_MODEL(img8, imgsz=768, conf=conf_thres, retina_masks=True, verbose=False)[0]
    masks = res.masks.data  # Tensor (N, H, W)

    h, w = input_image.shape
    if masks is None or masks.shape[0] == 0:
        binary = np.zeros((h, w), dtype=bool)
        labels = np.zeros((h, w), dtype=int)
    else:
        mb     = masks.bool().cpu()
        binary = torch.any(mb, dim=0).numpy()
        labels = np.zeros((h, w), dtype=int)
        for i in range(mb.shape[0]):
            labels[ mb[i].numpy() ] = i + 1

    if splitting:
        dist      = ndi.distance_transform_edt(binary)
        local_max = ndi.maximum_filter(dist, size=5) == dist
        markers, _= ndi.label(local_max)
        split     = watershed(-dist, markers, mask=binary)
        binary    = split > 0
        labels, _ = ndi.label(binary)

    return labels, binary


# --------------------------------------------------------------------
# core labeling function
# --------------------------------------------------------------------
def label_nuclei(
    dapi_image,
    *,
    method="otsu",               # choose "otsu" or "yolo"
    splitting=True,
    remove_small_outliers=False,
    remove_large_outliers=False,
    return_binary=False,
    initial_clipLimit=2.0,
    min_label_area=250,
    max_clip_iterations=10,
    max_baseline_size=80_000,
    verbose=False
):
    """
    dapi_image : 2D ndarray

    method : str
      "otsu" to use classic Otsu thresholding
      "yolo" to use the YOLO-v8 retina_masks segmentation

    The rest of the parameters control the CLAHE contrast-loop,
    optional splitting, outlier filtering, and min-area filtering.

    Returns:
      final_labels : labeled ndarray
      label_stats  : dict from cle.statistics_of_labelled_pixels
      (optional final_binary mask)
    """
    # pick the pipeline
    if method == "otsu":
        seg_fn = segmentation_pipeline_otsu
        seg_kwargs = {"splitting": False}
    elif method == "yolo":
        seg_fn = segmentation_pipeline_yolo
        seg_kwargs = {"splitting": False, "conf_thres": 0.1}
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'otsu' or 'yolo'")

    cle.select_device("cupy")
    img = np.copy(dapi_image)

    # STEP 1: baseline mask (no splitting in seg_fn call)
    base_labels, base_binary = seg_fn(img, **seg_kwargs)
    cc, _        = ndi.label(base_binary)
    sizes        = np.bincount(cc.ravel())[1:]
    baseline_max = sizes.max() if sizes.size else 0
    if verbose:
        print(f"[Baseline] largest = {baseline_max}")

    # CLAHE loop
    clip       = initial_clipLimit
    best_img   = img
    best_max   = baseline_max
    accepted   = img
    iters      = 0

    while iters < max_clip_iterations:
        clahe_img = cv2.createCLAHE(
            clipLimit=clip, tileGridSize=(8,8)
        ).apply(img.astype(np.uint8))

        # re-evaluate with same segmentation method
        _, bin_tmp = seg_fn(clahe_img, **seg_kwargs)
        cc_tmp, _  = ndi.label(bin_tmp)
        sizes_tmp  = np.bincount(cc_tmp.ravel())[1:]
        max_tmp    = sizes_tmp.max() if sizes_tmp.size else 0

        if verbose:
            print(f" iter {iters:2d}: clip={clip:.1f}  largest={max_tmp}")

        if max_tmp < best_max:
            best_max, best_img = max_tmp, clahe_img
        if max_tmp <= max_baseline_size:
            accepted = clahe_img
            if verbose: print(" → accepted (≤ max_baseline_size)")
            break

        if clip <= 0.2:
            break
        clip = round(max(clip - 0.2, 0.2), 2)
        iters += 1
    else:
        accepted = best_img if best_max < baseline_max else img
        if verbose:
            print(" CLAHE loop ended; using best-seen or baseline")

    # STEP 2: final segmentation (with splitting if requested)
    # allow splitting now
    if method == "otsu":
        final_labels, final_binary = segmentation_pipeline_otsu(
            accepted, splitting=splitting
        )
    else:
        final_labels, final_binary = segmentation_pipeline_yolo(
            accepted, splitting=splitting, conf_thres=seg_kwargs["conf_thres"]
        )

    # outlier + min-area filters
    arr = final_labels.get() if hasattr(final_labels, "get") else final_labels
    areas = np.bincount(arr.ravel())[1:]
    if areas.size:
        Q1, Q3 = np.percentile(areas, [25,75]); IQR = Q3 - Q1
        low, high = Q1 - 0.5*IQR, Q3 + 0.5*IQR
        filt = arr.copy()
        for lid, a in enumerate(areas, 1):
            if (remove_small_outliers and a < low) or \
               (remove_large_outliers and a > high) or \
               (a < min_label_area):
                filt[arr == lid] = 0
        final_labels = cle.relabel_sequential(
            cp.asarray(filt) if hasattr(final_labels, "get") else filt
        )

    stats = cle.statistics_of_labelled_pixels(dapi_image, final_labels)

    if return_binary:
        return final_labels, stats, final_binary
    return final_labels, stats

