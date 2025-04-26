#!/usr/bin/env python3
"""
validate_segmentation_matplotlib.py
-----------------------------------
Loads ND2 images, segments & classifies nuclei,
and shows for each image:

    • left panel  : DAPI  + classification/labels/binary
    • right panel : FITC  + classification
    • slider      : choose which image
    • checkboxes  : toggle Classification / Labels / Binary
"""

import os, sys, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# ─────────────────────────────────────────────────────────────
# CONFIG
sample_size   = 3
zoom          = '10'         # magnification filter, None for all
method        = 'yolo'       # 'yolo' | 'otsu'
input_folder  = r"G:/My Drive/KatzLab/TUNEL staining/Caitlin's Files/Raw nd2 Images"
kernel_size   = 51
confidenceThreshold = 1.0
# ─────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", message=".*siphash24.*")
warnings.filterwarnings("ignore",
    message=r".*clesperanto's cupy / CUDA backend is experimental.*",
    category=UserWarning)

module_dir = r"C:/VScode/TUNEL/"
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from tunel_quant import labeling, local_io, preprocessing, processing

status_to_rgba = {
    "definitely alive": (0.0, 1.0, 0.0, 1.0),
    "likely alive":     (0.4, 1.0, 0.4, 1.0),
    "likely dead":      (1.0, 0.55, 0.0, 1.0),
    "definitely dead":  (1.0, 0.0, 0.0, 1.0),
}

# ─────────────────────────────────────────────────────────────
def load_and_process(folder, n_images):
    imgs = local_io.pull_nd2_images(folder)
    if zoom:
        imgs = [img for img in imgs if f"{zoom}x".lower() in img[0].lower()]
    if n_images and n_images < len(imgs):
        imgs = random.sample(imgs, n_images)

    processed = []
    for name, dapi, fitc in imgs:
        prepd = preprocessing.preprocess_dapi(dapi)
        labels, _, binary = labeling.label_nuclei(
            prepd, method=method, return_binary=True, iterate=False,
            remove_large_outliers=False, remove_small_outliers=False
        )

        df = processing.analyze_nuclei(
            labels, fitc,
            kernel_size=kernel_size,
            confidenceThreshold=confidenceThreshold
        )
        class_map = dict(zip(df['nucleus_id'], df['alive_or_dead']))

        # fast LUT → classification
        max_id = int(labels.max())
        lut = np.ones((max_id+1, 4), dtype=np.float32)
        lut[:] = (1,1,1,0)        # transparent default
        for lid, status in class_map.items():
            lut[int(lid)] = status_to_rgba[status]
        classification = lut[labels]      # (H,W,4)

        processed.append(dict(
            name   = name,
            dapi   = dapi,
            fitc   = fitc,
            labels = labels,
            binary = binary.astype(np.uint8),
            classification= classification
        ))
    return processed
# ─────────────────────────────────────────────────────────────
def main():
    proc = load_and_process(input_folder, sample_size)
    if not proc:
        print("No images found."); return

    # ── initial figure with two panels ───────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    idx = 0
    # left (DAPI)
    raw1      = ax1.imshow(proc[idx]['dapi'], cmap='gray')
    classification1  = ax1.imshow(proc[idx]['classification'], interpolation='none')
    seg1      = ax1.imshow(proc[idx]['labels'], cmap='viridis', alpha=0.5)
    bin1      = ax1.imshow(proc[idx]['binary'], cmap='gray', alpha=0.3)
    ax1.set_title(f"{proc[idx]['name']} | DAPI"); ax1.axis('off')

    # right (FITC)  ⊕
    raw2      = ax2.imshow(proc[idx]['fitc'], cmap='gray')
    classification2  = ax2.imshow(proc[idx]['classification'], interpolation='none')
    ax2.set_title(f"{proc[idx]['name']} | FITC"); ax2.axis('off')

    # ── slider ───────────────────────────────────────────────
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider    = Slider(ax_slider, 'Image', 0, len(proc)-1,
                       valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        raw1.set_data(proc[i]['dapi'])
        classification1.set_data(proc[i]['classification'])
        seg1.set_data(proc[i]['labels'])
        bin1.set_data(proc[i]['binary'])
        ax1.set_title(f"{proc[i]['name']} | DAPI")

        raw2.set_data(proc[i]['fitc'])
        classification2.set_data(proc[i]['classification'])
        ax2.set_title(f"{proc[i]['name']} | FITC")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # ── checkboxes ───────────────────────────────────────────
    ax_check = plt.axes([0.02, 0.4, 0.18, 0.15])
    cb = CheckButtons(ax_check,
                      ['Classification','Labels','Binary'],
                      [True, True, True])

    def toggle(label):
        if label == 'Classification':
            classification1.set_visible(not classification1.get_visible())
            classification2.set_visible(not classification2.get_visible())   # ⊕ control both
        elif label == 'Labels':
            seg1.set_visible(not seg1.get_visible())
        else:  # 'Binary'
            bin1.set_visible(not bin1.get_visible())
        fig.canvas.draw_idle()

    cb.on_clicked(toggle)
    plt.show()
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

