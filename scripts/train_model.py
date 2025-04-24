import sys
import multiprocessing
import pathlib

# ─── determine project root (two levels up from scripts/) ──────────
HERE = pathlib.Path(__file__).resolve()           # .../TUNEL/scripts/train_model.py
ROOT = HERE.parent.parent                         # .../TUNEL

# now build all data‐paths off ROOT
DSB_RAW = ROOT / "yolo" / "dsb18" / "train_raw"
YOLO_DST = ROOT / "yolo" / "nuclei_yolo"

def main():
    # Make package importable
    module_dir = r"C:/VScode/TUNEL/"
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    import time
    from tunel_quant import yolo_model_training

    # Train the model
    start_time = time.time()

    print("Training YOLO model...")
    yolo_model_training.main()
    print("YOLO model training complete.")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f} seconds")
pass

if __name__ == "__main__":
    # this allows the spawn‐based loader to import safely
    multiprocessing.freeze_support()
    main()
    
