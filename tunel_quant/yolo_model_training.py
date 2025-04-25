"""Utility functions to build a DSB‑2018 nuclei dataset and fine‑tune YOLOv8.
All work is expressed as import‑safe functions so that other scripts can orchestrate
execution without side‑effects.

Suggested layout:
TUNEL/
 ├─ tunel_quant/
 │   ├─ __init__.py
 │   └─ tunel_model_training.py   ← (this file)
 └─ scripts/
     └─ train_model.py            ← driver script
"""
from __future__ import annotations
import os, zipfile, shutil, random, pathlib, sys
import cv2
import numpy as np
from ultralytics import YOLO

# ─── project paths ──────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
RAW_DIR   = ROOT / "yolo" / "dsb18" / "train_raw"
YOLO_DIR  = ROOT / "yolo" / "nuclei_yolo"
DATA_YAML = YOLO_DIR / "data.yaml"

# Ensure "tunel_quant.preprocessing" is importable when module is run as a script
MODULE_DIR = str(ROOT)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from tunel_quant import preprocessing   # preprocess_dapi(img)

# Kaggle creds (edit as needed)
KAG_DIR = pathlib.Path(r"C:/Users/zbedd/OneDrive/GT/Opportunities/Katz Lab")
os.environ["KAGGLE_CONFIG_DIR"] = str(KAG_DIR)
from kaggle.api.kaggle_api_extended import KaggleApi

# ─── dataset helpers ────────────────────────────────────────────────────────

def download_dsb() -> None:
    """Download Data‑Science‑Bowl‑2018 if missing and unzip under RAW_DIR."""
    target = RAW_DIR.parent
    target.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    api.competition_download_file(
        "data-science-bowl-2018", "stage1_train.zip", path=str(target), quiet=False
    )
    with zipfile.ZipFile(target / "stage1_train.zip") as zf:
        zf.extractall(target / "train_raw")
    print("✅ DSB‑2018 extracted →", target / "train_raw")


def masks_to_poly(mask_dir: pathlib.Path, lbl_txt: pathlib.Path) -> None:
    """Convert binary mask PNGs into YOLO polygon txt."""
    lbl_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_txt, "w") as f:
        for m in mask_dir.glob("*.png"):
            img = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c = c.squeeze()
                if c.ndim != 2 or c.shape[0] < 3:
                    continue
                xs, ys = c[:, 0] / w, c[:, 1] / h
                poly = " ".join(map(str, np.stack([xs, ys], 1).reshape(-1)))
                f.write(f"0 {poly}\n")


def build_dataset(train_blur_ratio: float = 0.7) -> None:
    """Create YOLO‑seg folder‑tree and YAML. Safe to call repeatedly."""
    if not RAW_DIR.exists():
        download_dsb()

    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    train_ids = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
    val_ids = {train_ids[i] for i in range(0, len(train_ids), 10)}  # 10 % val

    for sid in train_ids:
        split = "val" if sid in val_ids else "train"
        img_src = RAW_DIR / sid / "images" / f"{sid}.png"
        img_dst = YOLO_DIR / "images" / split / f"{sid}.png"
        lbl_dst = YOLO_DIR / "labels" / split / f"{sid}.txt"

        img = cv2.imread(str(img_src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("⚠️  skip", img_src)
            continue

        if split == "train" and random.random() < train_blur_ratio:
            img = preprocessing.preprocess_dapi(img)

        cv2.imwrite(str(img_dst), img)
        masks_to_poly(RAW_DIR / sid / "masks", lbl_dst)

    with open(DATA_YAML, "w") as f:
        f.write(
            f"path: {YOLO_DIR.resolve()}\n"
            "train: images/train\n"
            "val:   images/val\n"
            "nc: 1\n"
            "names: ['nucleus']\n"
        )
    print("✅ YOLO‑seg dataset ready →", YOLO_DIR)

# ─── model training ─────────────────────────────────────────────────────────

def train_yolov8(
    epochs: int = 50,
    imgsz: int = 768,
    batch: int = 8,
    device: int | str = 0,
) -> None:
    """Fine‑tune `yolov8m‑seg.pt` on the prepared dataset."""
    build_dataset()

    print("▶️  fine‑tuning yolov8m‑seg …")
    model = YOLO("yolov8m-seg.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        mask_ratio=2,
        retina_masks=True,
        device=device,
    )
    print("✅ training complete.  Checkpoints under runs/segment/")

# Optional: allow this file to be run directly for quick tests
if __name__ == "__main__":
    train_yolov8()