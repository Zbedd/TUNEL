"""CLI entry‑point to train the YOLO nuclei‑segmentation model."""
import multiprocessing
import time

# Import YOLO functionality from external package
from imageProcessingUtils.yolo import build_dataset, train_yolov8


def main():
    start = time.time()

    # 1️⃣ build the dataset (no‑op if already built)
    build_dataset()

    # 2️⃣ train the model
    train_yolov8()

    print(f"Total elapsed: {time.time() - start:.1f} s")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # for Windows spawn
    main()
