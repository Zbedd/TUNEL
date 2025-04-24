'''Preprocessing functions should accept and return a dapi image'''

import cv2
import numpy as np

def preprocess_dapi(image):
    """
    Preprocess a single‐channel DAPI image to reduce interior grain and boost edge contrast.

    Pipeline:
      1. Normalize to 0–255 and convert to uint8.
      2. Median blur (ksize=5) to remove speckle noise.
      3. Bilateral filter (d=9, σColor=75, σSpace=75) for edge‐preserving smoothing.
      4. CLAHE (clipLimit=2.0, tileGridSize=8×8) for local contrast enhancement.

    Parameters
    ----------
    image : ndarray
        2D array (any numeric type) holding the raw DAPI intensities.

    Returns
    -------
    proc : ndarray, uint8
        The preprocessed image, ready for downstream segmentation.
    """
    # 1) normalize to [0,255]
    img = image.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img.astype(np.uint8)

    # 2) median filter to knock down speckles
    med = cv2.medianBlur(img_uint8, ksize=5)

    # 3) bilateral filter for edge‐aware smoothing
    #    (d=9 picks a ~9×9 neighborhood)
    bilat = cv2.bilateralFilter(med, d=9, sigmaColor=75, sigmaSpace=75)

    # 4) CLAHE to boost local contrast without blowing out edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc = clahe.apply(bilat)

    return proc