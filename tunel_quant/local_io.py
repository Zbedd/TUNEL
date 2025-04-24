#Imports
import os
import numpy as np
import cv2
import nd2reader

def pull_nd2_images(folder):
    """
    Pull ND2 images from a folder and return a list where each element
    is a three-element list [Name, DAPI, FITC] representing the two channels.
    The two channels will be returned as uint8. Name comes from the file name.

    Parameters:
      folder (str): Path to the folder containing the ND2 files.

    Assumptions:
      - Each ND2 file contains a single multi-channel image.
      - The ND2 file has two channels.
      - The channel names are stored in nd2.metadata['channels'].
      - Channel 0 must be 'DAPI' and channel 1 must be 'FITC' (case-insensitive).

    Raises:
      ValueError: If channel metadata is missing or if the channel names do not match.
    """
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(".nd2"):
            file_path = os.path.join(folder, file)
            with nd2reader.ND2Reader(file_path) as nd2:
                # Bundle axes so that the image is returned with dimensions (channels, height, width)
                nd2.bundle_axes = 'cyx'

                # Retrieve channel names from metadata; throw an error if not present.
                try:
                    channels = nd2.metadata['channels']
                except KeyError:
                    raise ValueError(f"File {file} does not contain channel metadata in nd2.metadata['channels']")

                if len(channels) != 2:
                    raise ValueError(f"File {file} has {len(channels)} channels, expected 2 channels.")
                if channels[0].upper() != "DAPI":
                    raise ValueError(f"File {file} channel 0 is not DAPI, found '{channels[0]}'.")
                if channels[1].upper() != "FITC":
                    raise ValueError(f"File {file} channel 1 is not FITC, found '{channels[1]}'.")

                # Retrieve the first frame (assuming one multi-channel image per file)
                image = nd2[0]

                # Check that the image shape is as expected: (channels, height, width)
                if image.ndim == 3 and image.shape[0] == 2:
                    dapi = image[0]
                    dapi_uint8 = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    fitc = image[1]
                    fitc_uint8 = cv2.normalize(fitc, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    images.append([str(file), dapi_uint8, fitc_uint8])
                else:
                    raise ValueError(
                        f"File {file} has unexpected shape {image.shape}. Expected a 2-channel image with shape (2, height, width)."
                    )
    return images

def uint16_to_float(image):
  return (image/65536).astype(np.float32)

def uint16_to_uint8(image):
  return (image/65536*255).astype(np.uint8)