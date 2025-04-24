def main():
    from ultralytics import YOLO
    import pathlib, shutil, os, cv2
    import numpy as np
    import scipy.ndimage as ndi
    import zipfile

    #Path to kaggle.json file: C:\Users\zbedd\OneDrive\GT\Opportunities\Katz Lab\kaggle.json

    #  ➤ 1) Configure Kaggle API
    # ----------------------------------------
    # Change this to the folder containing your kaggle.json
    os.environ["KAGGLE_CONFIG_DIR"] = r"C:/Users/zbedd/OneDrive/GT/Opportunities/Katz Lab/"

    from kaggle.api.kaggle_api_extended import KaggleApi

    def download_dsb2018(target="dsb18"):
        target = pathlib.Path(target)
        target.mkdir(parents = True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        # download only the training zip
        api.competition_download_file(
            "data-science-bowl-2018",
            file_name="stage1_train.zip",
            path=str(target),
            quiet=False
        )
        # unzip it
        with zipfile.ZipFile(target/"stage1_train.zip", "r") as zf:
            zf.extractall(target/"train_raw")
        print("✅ Downloaded and unpacked DSB2018 stage1_train.zip into", target/"train_raw")
        
    # Check if yolo_wrapper/dsb18/train_raw exists, if not, download the dataset
    if not pathlib.Path("yolo/dsb18/train_raw").exists():
        download_dsb2018(target="yolo/dsb18")
        

  
    #  ➤ 2) Prepare YOLO dataset
    # ----------------------------------------
    src = pathlib.Path("yolo/dsb18/train_raw")
    dst = pathlib.Path("yolo/nuclei_yolo")
    # create folder structure
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (dst/sub).mkdir(parents=True, exist_ok=True)

    # split 10% for validation
    train_ids = sorted([p.name for p in src.iterdir() if p.is_dir()])
    val_ids   = set(train_ids[i] for i in range(0, len(train_ids), 10))

    def masks_to_yolo_poly(mask_dir, y_txt):
        """
        Reads all *.png masks in mask_dir and writes a single YOLO-seg .txt with
        polygon format: class_id x1 y1 x2 y2 ... (normalized 0–1).
        """
        mask_dir = pathlib.Path(mask_dir)
        with open(y_txt, "w") as out:
            for m in mask_dir.glob("*.png"):
                im = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                h, w = im.shape
                cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    cnt = cnt.squeeze()
                    if cnt.ndim != 2 or cnt.shape[0] < 3:
                        continue
                    xs = cnt[:,0] / w
                    ys = cnt[:,1] / h
                    coords = np.stack([xs, ys], axis=1).reshape(-1)
                    poly = " ".join(map(str, coords))
                    out.write(f"0 {poly}\n")

    # copy images & generate labels
    for sid in train_ids:
        split = "val" if sid in val_ids else "train"
        img_src = src/sid/"images"/f"{sid}.png"
        m_dir   = src/sid/"masks"
        img_dst = dst/"images"/split/f"{sid}.png"
        lbl_dst = dst/"labels"/split/f"{sid}.txt"
        shutil.copy(img_src, img_dst)
        masks_to_yolo_poly(m_dir, lbl_dst)
    print("✅ Prepared YOLO dataset in", dst)



    #  ➤ 3) Write data.yaml
    # ----------------------------------------
    data_yaml = dst/"data.yaml"
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {dst.resolve()}\n"
            "train: images/train\n"
            "val:   images/val\n"
            "nc:    1\n"
            "names: ['nucleus']\n"
        )
    print("✅ Wrote", data_yaml)

    #  ➤ 4) Fine-tune YOLO-v8 segmentation
    # ----------------------------------------
    # install ultralytics if needed:
    # pip install ultralytics
    from ultralytics import YOLO

    print("▶️ Starting YOLO-v8n segmentation training…")
    model = YOLO("yolov8n-seg.pt")  # pretrained COCO-seg checkpoint
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=512,
        batch=8,
        device=0  # GPU
    )
    print("✅ Training complete. Checkpoint at runs/segment/train*/weights/best.pt")

    pass

if __name__ == "__main__":
    main()