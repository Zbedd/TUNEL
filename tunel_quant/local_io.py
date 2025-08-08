"""
Local I/O Operations for Microscopy Image Data

This module handles input/output operations specific to microscopy data formats,
particularly Nikon ND2 files containing multi-channel fluorescence images.

Key functionality:
- Import ND2 files with DAPI/FITC channel validation
- Automatic intensity normalization and data type conversion
- Batch processing of microscopy image folders
- Utility functions for common data type conversions

The module is designed for TUNEL assay workflows where DAPI and FITC channels
need to be extracted and processed from proprietary microscopy file formats.
"""

# Imports
import os
import numpy as np
import cv2
import nd2reader

def pull_nd2_images(folder):
    """
    Extract DAPI and FITC channels from ND2 files in a specified folder.
    
    This function processes Nikon ND2 microscopy files containing dual-channel
    fluorescence images for TUNEL assay analysis. Each file is expected to contain
    exactly two channels: DAPI (nuclei) and FITC (cell death marker).

    The function performs automatic intensity normalization to maximize dynamic range
    and converts all images to uint8 format for consistent downstream processing.

    Parameters
    ----------
    folder : str
        Path to directory containing ND2 files to process.

    Returns
    -------
    list of list
        List where each element is [filename, dapi_image, fitc_image].
        - filename (str): Original ND2 filename  
        - dapi_image (ndarray): DAPI channel as normalized uint8 array
        - fitc_image (ndarray): FITC channel as normalized uint8 array

    Raises
    ------
    ValueError
        If channel metadata is missing, incorrect number of channels found,
        or channel names don't match expected 'DAPI' and 'FITC' (case-insensitive).
        Also raised if image dimensions are unexpected.

    Notes
    -----
    - Assumes each ND2 file contains a single multi-channel image
    - Channel validation is case-insensitive  
    - Images are normalized using cv2.NORM_MINMAX to utilize full 0-255 range
    - Processing order: channel 0 = DAPI, channel 1 = FITC
    """
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(".nd2"):
            file_path = os.path.join(folder, file)
            with nd2reader.ND2Reader(file_path) as nd2:
                # Configure axes bundling for (channels, height, width) format
                nd2.bundle_axes = 'cyx'

                # Validate channel metadata exists and extract channel names
                try:
                    channels = nd2.metadata['channels']
                except KeyError:
                    raise ValueError(f"File {file} does not contain channel metadata in nd2.metadata['channels']")

                # Validate exactly 2 channels present
                if len(channels) != 2:
                    raise ValueError(f"File {file} has {len(channels)} channels, expected 2 channels.")
                
                # Validate channel names match expected DAPI/FITC pattern
                if channels[0].upper() != "DAPI":
                    raise ValueError(f"File {file} channel 0 is not DAPI, found '{channels[0]}'.")
                if channels[1].upper() != "FITC":
                    raise ValueError(f"File {file} channel 1 is not FITC, found '{channels[1]}'.")

                # Extract first frame (assuming single timepoint per file)
                image = nd2[0]

                # Validate image dimensions and extract channels
                if image.ndim == 3 and image.shape[0] == 2:
                    # Extract and normalize DAPI channel (channel 0)
                    dapi = image[0]
                    dapi_uint8 = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Extract and normalize FITC channel (channel 1)  
                    fitc = image[1]
                    fitc_uint8 = cv2.normalize(fitc, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    images.append([str(file), dapi_uint8, fitc_uint8])
                else:
                    raise ValueError(
                        f"File {file} has unexpected shape {image.shape}. Expected a 2-channel image with shape (2, height, width)."
                    )
    return images


def uint16_to_float(image):
    """
    Convert 16-bit unsigned integer image to 32-bit float in range [0, 1].
    
    Parameters
    ----------
    image : ndarray
        Input image with uint16 data type.
        
    Returns
    -------
    ndarray
        Image converted to float32 with values normalized to [0, 1] range.
    """
    return (image / 65536).astype(np.float32)


def uint16_to_uint8(image):
    """
    Convert 16-bit unsigned integer image to 8-bit unsigned integer.
    
    This function performs linear scaling from 16-bit range [0, 65535] 
    to 8-bit range [0, 255].
    
    Parameters
    ----------
    image : ndarray
        Input image with uint16 data type.
        
    Returns
    -------
    ndarray
        Image converted to uint8 with values scaled to [0, 255] range.
    """
    return (image / 65536 * 255).astype(np.uint8)