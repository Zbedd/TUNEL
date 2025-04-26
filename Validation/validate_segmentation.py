#!/usr/bin/env python3
"""
validate_segmentation_matplotlib.py
-----------------------------------
Loads your ND2 images, segments & classifies nuclei,
and displays them in one Matplotlib window with:

  • Slider to pick which image to view
  • Checkboxes to toggle:
      – alive/dead overlay (green↔red)
      – segmentation labels
      – binary mask
"""

import os

# force TkAgg to avoid Qt‐plugin issues
import matplotlib
matplotlib.use('TkAgg')
import cv2

import sys, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak before running
sample_size         = 3
method              = 'yolo'  # 'yolo' or 'otsu'
input_folder        = r"G:/My Drive/KatzLab/TUNEL staining/Caitlin's Files/Raw nd2 Images"
kernel_size         = 51      # for analyze_nuclei background estimation
confidenceThreshold = 1.0     # for analyze_nuclei
# ──────────────────────────────────────────────────────────────────────────────

# silence spurious warnings
warnings.filterwarnings("ignore", message=".*siphash24.*")
warnings.filterwarnings(
    "ignore",
    message=r".*clesperanto's cupy / CUDA backend is experimental.*",
    category=UserWarning
)

# make your package importable
module_dir = r"C:/VScode/TUNEL/"
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from tunel_quant import labeling, local_io, preprocessing, processing

def load_and_process(folder, n_images, seg_method, kernel_size, confidenceThreshold):
    """
    Returns list of dicts, each with keys:
      'name','dapi','labels','binary','overlay'
    where 'overlay' is an (H,W,4) float32 RGBA array.
    """
    imgs = local_io.pull_nd2_images(folder)
    if n_images and n_images < len(imgs):
        imgs = random.sample(imgs, n_images)

    status_to_rgba = {
        "definitely alive": (0.0, 1.0, 0.0, 1.0),
        "likely alive":     (0.4, 1.0, 0.4, 1.0),
        "likely dead":      (1.0, 0.55, 0.0, 1.0),
        "definitely dead":  (1.0, 0.0, 0.0, 1.0),
    }

    processed = []
    for name, dapi, fitc in imgs:
        # segment
        prep = preprocessing.preprocess_dapi(dapi)
        labels, _, binary = labeling.label_nuclei(
            prep,
            remove_large_outliers=False,
            remove_small_outliers=False,
            method=seg_method,
            return_binary=True,
            iterate=False
        )
        # classify
        df = processing.analyze_nuclei(
            labels, fitc,
            kernel_size=kernel_size,
            confidenceThreshold=confidenceThreshold
        )
        class_map = dict(zip(df['nucleus_id'], df['alive_or_dead']))
        # build RGBA overlay

        print('starting')
        h, w = labels.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)

        # 1) invert map: status -> list of label_ids
        from collections import defaultdict
        status_to_ids = defaultdict(list)
        for lid, status in class_map.items():
            status_to_ids[status].append(lid)
            

        # 2) one boolean mask per status
        for status, lids in status_to_ids.items():
            if not lids: 
                continue
            mask = np.isin(labels, lids)       # single pass
            overlay[mask] = status_to_rgba[status]
    
        processed.append({
            'name':   name,
            'dapi':   dapi,
            'labels': labels,
            'binary': binary.astype(np.uint8),
            'overlay': overlay
        })
    return processed

def main():
    proc = load_and_process(
        input_folder,
        sample_size,
        method,
        kernel_size,
        confidenceThreshold
    )
    N = len(proc)
    if N == 0:
        print("No images found.")
        return

    # initial index
    idx = 0

    # set up figure + main axes
    fig, ax = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # display initial images
    raw_im = ax.imshow(proc[idx]['dapi'], cmap='gray')
    overlay_im = ax.imshow(proc[idx]['overlay'], interpolation='none', alpha=1.0)
    seg_im = ax.imshow(proc[idx]['labels'], cmap='viridis', alpha=0.5)
    bin_im = ax.imshow(proc[idx]['binary'], cmap='gray', alpha=0.3)

    ax.set_title(proc[idx]['name'])
    ax.axis('off')

    # Slider for image index
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Image', 0, N-1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        raw_im.set_data(proc[i]['dapi'])
        overlay_im.set_data(proc[i]['overlay'])
        seg_im.set_data(proc[i]['labels'])
        bin_im.set_data(proc[i]['binary'])
        ax.set_title(proc[i]['name'])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # CheckButtons for layers
    ax_check = plt.axes([0.02, 0.4, 0.18, 0.15])
    labels = ['Overlay','Labels','Binary']
    actives = [True, True, True]
    check = CheckButtons(ax_check, labels, actives)

    def toggle(label):
        if label == 'Overlay':
            overlay_im.set_visible(not overlay_im.get_visible())
        elif label == 'Labels':
            seg_im.set_visible(not seg_im.get_visible())
        else:  # 'Binary'
            bin_im.set_visible(not bin_im.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    plt.show()

if __name__ == "__main__":
    main()