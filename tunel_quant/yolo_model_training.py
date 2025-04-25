"""
Fine-tune YOLO-v8m on the DSB-2018 nuclei set, adding a run-time
“DAPI-blur” transform that mimics bright-edge / spotty-centre nuclei.

 ─ pipeline ─
 1) download & unzip Kaggle DSB 2018 (if absent)
 2) convert instance masks → YOLO-seg polygons
 3) write data.yaml
 4) train yolov8m-seg with custom transform (calls preprocessing on p-0.7 of images per epoch)
"""

from __future__ import annotations
import os, random, zipfile, shutil, pathlib, cv2
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from ultralytics import YOLO
from ultralytics.data.augment import Compose
import random
from preprocessing import preprocess_dapi     


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG ─ adjust these paths if needed
# ──────────────────────────────────────────────────────────────────────────────
KAGGLE_JSON  = r"C:/Users/zbedd/OneDrive/GT/Opportunities/Katz Lab/kaggle.json"
RAW_DIR      = pathlib.Path("yolo/dsb18/train_raw")
YOLO_DIR     = pathlib.Path("yolo/nuclei_yolo")          # final dataset root
WEIGHTS_OUT  = "runs/segment"                            # Ultralytics default

# ensure Kaggle token visible for kaggle-api
os.environ["KAGGLE_CONFIG_DIR"] = str(pathlib.Path(KAGGLE_JSON).parent)


# ──────────────────────────────────────────────────────────────────────────────
# util: download stage1_train.zip once
# ──────────────────────────────────────────────────────────────────────────────
def download_dsb2018() -> None:
    target = RAW_DIR.parent        # yolo/dsb18
    target.mkdir(parents=True, exist_ok=True)

    api = KaggleApi(); api.authenticate()
    api.competition_download_file(
        "data-science-bowl-2018",
        file_name="stage1_train.zip",
        path=str(target),
        quiet=False,
    )
    with zipfile.ZipFile(target/"stage1_train.zip", "r") as zf:
        zf.extractall(target/"train_raw")
    print("✅ DSB2018 downloaded and extracted →", target/"train_raw")


# ──────────────────────────────────────────────────────────────────────────────
# util: convert per-instance PNG masks → one YOLO polygon .txt
# ──────────────────────────────────────────────────────────────────────────────
def masks_to_yolo_poly(mask_dir: pathlib.Path, y_txt: pathlib.Path) -> None:
    with open(y_txt, "w") as out:
        for m_path in mask_dir.glob("*.png"):
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is None: continue
            h, w = m.shape
            cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c = c.squeeze()
                if c.ndim != 2 or c.shape[0] < 3: continue
                xs, ys = c[:,0]/w, c[:,1]/h
                poly   = " ".join(map(str, np.stack([xs,ys],1).reshape(-1)))
                out.write(f"0 {poly}\n")


# ──────────────────────────────────────────────────────────────────────────────
# on-the-fly blur transform (applied only to training batches)
# ──────────────────────────────────────────────────────────────────────────────
def blur_transform(data, p: float = 0.7):
    """
    Apply preprocess_dapi to the W×H×3 training image with prob *p*.
    Ultralytics passes BGR uint8; we convert to 1-ch, blur, stack back to 3-ch.
    """
    if random.random() > p:
        return data                            # leave raw 30 % of the time

    img = data["img"]                         # BGR uint8, shape (H,W,3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_dapi(gray)              # <- your pipeline
    data["img"] = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    return data



# ──────────────────────────────────────────────────────────────────────────────
# STEP-0: fetch dataset if missing
# ──────────────────────────────────────────────────────────────────────────────
if not RAW_DIR.exists():
    download_dsb2018()


# ──────────────────────────────────────────────────────────────────────────────
# STEP-1: build yolo-seg folder structure (raw images; labels polygons)
# ──────────────────────────────────────────────────────────────────────────────
for sub in ("images/train", "labels/train", "images/val", "labels/val"):
    (YOLO_DIR/sub).mkdir(parents=True, exist_ok=True)

train_ids = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
val_ids   = {train_ids[i] for i in range(0, len(train_ids), 10)}   # 10 % val

for sid in train_ids:
    split    = "val" if sid in val_ids else "train"
    img_src  = RAW_DIR/sid/"images"/f"{sid}.png"
    m_dir    = RAW_DIR/sid/"masks"
    img_dst  = YOLO_DIR/"images"/split/f"{sid}.png"
    lbl_dst  = YOLO_DIR/"labels"/split/f"{sid}.txt"

    shutil.copy(img_src, img_dst)        # keep RAW copy; blur happens on-the-fly
    masks_to_yolo_poly(m_dir, lbl_dst)
print("✅ YOLO dataset ready:", YOLO_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# STEP-2: write data.yaml for Ultralytics
# ──────────────────────────────────────────────────────────────────────────────
data_yaml = YOLO_DIR/"data.yaml"
with open(data_yaml, "w") as f:
    f.write(
        f"path: {YOLO_DIR.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc:    1\n"
        "names: ['nucleus']\n"
    )
print("✅ Wrote", data_yaml)


# ──────────────────────────────────────────────────────────────────────────────
# STEP-3: fine-tune with custom transform
# ──────────────────────────────────────────────────────────────────────────────
print("▶️  training YOLO-v8m-seg with on-the-fly blur…")
model  = YOLO("yolov8m-seg.pt")
model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=768,
    batch=8,
    mask_ratio=2,
    retina_masks=True,
    device=0,
    transforms=Compose([blur_transform]),   # ← now using your preprocess
)
print("✅ done. best checkpoint in", WEIGHTS_OUT)