"""CLI entry‑point to train the YOLO nuclei‑segmentation model."""
import multiprocessing
import time
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../TUNEL
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from tunel_quant import yolo_model_training as tmt  # adjust the import path if moved


def main():
    start = time.time()

    # 1️⃣ build the dataset (no‑op if already built)
    tmt.build_dataset()

    # 2️⃣ train the model
    tmt.train_yolov8()

    print(f"Total elapsed: {time.time() - start:.1f} s")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # for Windows spawn
    main()