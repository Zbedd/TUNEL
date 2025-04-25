"""
Fine-tune YOLO-v8m on DSB-2018 nuclei with **offline** blur:

• 70 % of training PNGs are saved after preprocess_dapi()
• validation PNGs stay raw
• no transforms= / pre_transform= required
"""

from __future__ import annotations
import os, zipfile, shutil, random, pathlib, cv2, sys
import numpy as np
from ultralytics import YOLO

# ── import your blur pipeline ────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
MODULE_DIR = r"C:/VScode/TUNEL"
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from tunel_quant import preprocessing       # preprocess_dapi(img)

# ── paths ────────────────────────────────────────────────────────────────────
RAW_DIR   = ROOT / "yolo" / "dsb18" / "train_raw"
YOLO_DIR  = ROOT / "yolo" / "nuclei_yolo"
KAG_DIR   = pathlib.Path(r"C:/Users/zbedd/OneDrive/GT/Opportunities/Katz Lab")
os.environ["KAGGLE_CONFIG_DIR"] = str(KAG_DIR)
from kaggle.api.kaggle_api_extended import KaggleApi

# ── dataset download (once) ─────────────────────────────────────────────────
def download_dsb():
    target = RAW_DIR.parent
    target.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    api.competition_download_file("data-science-bowl-2018",
                                  "stage1_train.zip",
                                  path=str(target),
                                  quiet=False)
    with zipfile.ZipFile(target/"stage1_train.zip") as zf:
        zf.extractall(target/"train_raw")
    print("✅ DSB-2018 extracted →", target/"train_raw")

if not RAW_DIR.exists():
    download_dsb()

# ── util: masks → polygon txt ───────────────────────────────────────────────
def masks_to_poly(mask_dir: pathlib.Path, lbl_txt: pathlib.Path):
    lbl_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_txt, "w") as f:
        for m in mask_dir.glob("*.png"):
            img = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            h, w = img.shape
            cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                c = c.squeeze()
                if c.ndim != 2 or c.shape[0] < 3: continue
                xs, ys = c[:,0]/w, c[:,1]/h
                poly   = " ".join(map(str, np.stack([xs,ys],1).reshape(-1)))
                f.write(f"0 {poly}\n")

# ── build YOLO-seg folders with 70 % pre-blurred train PNGs ─────────────────
for sub in ("images/train","labels/train","images/val","labels/val"):
    (YOLO_DIR/sub).mkdir(parents=True, exist_ok=True)

train_ids = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
val_ids   = {train_ids[i] for i in range(0,len(train_ids),10)}   # 10 % val

for sid in train_ids:
    split   = "val" if sid in val_ids else "train"
    img_src = RAW_DIR/sid/"images"/f"{sid}.png"
    img_dst = YOLO_DIR/"images"/split/f"{sid}.png"
    lbl_dst = YOLO_DIR/"labels"/split/f"{sid}.txt"

    img = cv2.imread(str(img_src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("⚠️  skip", img_src); continue

    if split == "train" and random.random() < 0.7:        # 70 % blur
        img = preprocessing.preprocess_dapi(img)

    cv2.imwrite(str(img_dst), img)                        # save PNG
    masks_to_poly(RAW_DIR/sid/"masks", lbl_dst)

print("✅ YOLO-seg dataset with pre-blur ready:", YOLO_DIR)

# ── write data.yaml ─────────────────────────────────────────────────────────
DATA_YAML = YOLO_DIR/"data.yaml"
with open(DATA_YAML,"w") as f:
    f.write(
        f"path: {YOLO_DIR.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 1\n"
        "names: ['nucleus']\n"
    )

# ── train YOLO-v8m -- no custom transform needed ───────────────────────────
print("▶️  fine-tuning yolov8m-seg (pre-blur mode)…")
model = YOLO("yolov8m-seg.pt")
model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=768,
    batch=8,
    mask_ratio=2,
    retina_masks=True,
    device=0,
)
print("✅ training complete.  Checkpoints under runs/segment/")
