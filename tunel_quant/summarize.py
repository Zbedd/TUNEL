"""
Data Analysis Workflow Module for TUNEL Experiments

This module provides high-level analysis workflows that orchestrate the complete
TUNEL assay pipeline from raw ND2 images to statistical summaries. It handles
batch processing of microscopy data and integrates segmentation, classification,
and statistical analysis steps.

Key functionality:
- Batch processing of ND2 image folders
- Complete TUNEL analysis pipeline integration
- Experimental metadata handling and parsing
- Results aggregation and standardization
- Sex-based filtering and magnification selection

The module serves as the main entry point for complete TUNEL analysis workflows,
coordinating between segmentation, processing, and analysis modules.
"""

import pandas as pd
import time
from . import labeling, local_io, processing


def analyze_folder(path, apply_masks=False, mask_folder=None, sex=None, sex_path=None, method='otsu', conThresh=0.8, kSize=31, magnification=None):
    """
    Perform complete TUNEL analysis on all ND2 images in a folder.
    
    This function orchestrates the full analysis pipeline for a batch of ND2 images,
    including image loading, nuclear segmentation, cell death classification, and
    results aggregation. It handles experimental metadata extraction and provides
    flexible filtering options.
    
    The analysis workflow:
    1. Load ND2 images and extract DAPI/FITC channels
    2. Perform nuclear segmentation on DAPI images  
    3. Classify cell viability based on FITC fluorescence
    4. Apply optional spatial masking and filtering
    5. Aggregate results with experimental metadata

    Parameters
    ----------
    path : str
        Directory path containing ND2 image files for analysis.
    apply_masks : bool, default=False
        Whether to apply spatial region masks during analysis.
    mask_folder : str, optional
        Path to directory containing mask files. Required if apply_masks=True.
        Mask files should match image filenames with '_mask.tif' suffix.
    sex : str, optional
        Filter for specific mouse sex ('m' or 'f'). None includes all mice.
    sex_path : str, optional
        Path to CSV file with 'Mouse' and 'Sex' columns for sex filtering.
    method : str, default='otsu'
        Nuclear segmentation method. Options: 'otsu' or 'yolo'.
    conThresh : float, default=0.8
        Confidence threshold for alive/dead classification. Must be ≥ 1.0.
    kSize : int, default=31
        Kernel size for background estimation in cell analysis.
    magnification : int, optional
        Filter images by magnification level. None includes all magnifications.

    Returns
    -------
    list
        Analysis results as list of [image_name, analysis_dataframe] pairs.
        Each analysis_dataframe contains nucleus-level measurements with columns:
        - nucleus_id: Unique identifier for each nucleus
        - absolute_brightness: Raw FITC intensity  
        - relative_brightness: Background-corrected intensity
        - alive_or_dead: Classification string
        
    Notes
    -----
    - Automatically extracts experimental metadata from ND2 filenames
    - Handles missing files gracefully with warnings
    - Progress tracking for batch processing
    - Results compatible with downstream statistical analysis functions
    """
    all_analysis = []  # List to store analysis results for each image.

    # Load all ND2 images from the folder.
    images = local_io.pull_nd2_images(path)

    image_count = 0
    timestamp = time.time()
    
    if magnification:
        print(f"{len(images)} images found in {path}.")
        print(f"Filtering images by magnification: {magnification}x")
        images = [img for img in images if f"{magnification}x".lower() in img[0].lower()]
        print(f"{len(images)} images found with {magnification}x magnification.")
    else:
        print(f"Analyzing {len(images)} images...")
    
    if sex is not None:
        print(f"Filtering images by sex: " + str(sex))
        sex_csv = pd.read_csv(sex_path)
        mice = (
            sex_csv.loc[sex_csv['Sex'].str.lower() == sex.lower(), 'Mouse']
            .astype(str).str.strip()        # to be safe, strip whitespace
            .tolist()
        )        
        images = [img for img in images if img[0].split('_')[0] in mice]

    for image in images:
        # Unpack the image components:
        # image[0] -> name, image[1] -> DAPI channel image, image[2] -> FITC channel image.
        name = image[0]
        dapi = image[1]
        fitc = image[2]

        # Perform nuclear labeling on the DAPI image, with both large and small outliers removed.
        nucLabels, nucLabelStats = labeling.label_nuclei(dapi, remove_large_outliers=True, remove_small_outliers=True, method=method, apply_masking=apply_masks, mask_folder=mask_folder, name = name)

        # Analyze nuclei using the FITC image to compute brightness and viability.
        analysis = processing.analyze_nuclei(nucLabels, fitc, kernel_size = kSize, confidenceThreshold=conThresh)

        # Append the image name and its analysis result to the aggregate list.
        all_analysis.append([name, analysis])
        
        image_count += 1
        if image_count % 10 == 0:
            print(f"Processed {image_count} images...")
            elapsed_time = time.time() - timestamp
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return all_analysis

def summarize_analysis(all_analysis, location_map = None):
  '''
  Accepts the output of analyze_folder and returns a summary of the analysis in the
  form of a dataframe with columns ['name', 'group', 'location', 'mouse', 'definitely alive', 'definitely dead', 'likely alive', 'likely dead']
  
  Reformats on the analyze_folder dataframe with location, groups, and counts of alive/dead cells.
  '''

  group_map = {
      'ctrl_CRE+': ['ctrl_CRE+'],
      'ctrl_CRE-': ['ctrl_CRE-'],
      'PLX_CRE+': ['PLX_CRE+'],
      'PLX_CRE-': ['PLX_CRE-']
  }

  if location_map is None:
    location_map = {
        'cortex': ['CORTEX', 'cortex', 'corex'],
        'CA1': ['CA1', 'CA!'],
        'CA2': ['CA2'],
        'CA3': ['CA3'],
        'cpu': ['CPU', 'cpu'],
        'brainstem': ['brainstem'],
        'DG': ['DG'],
        'Anterior hippocampus': ['Anterior hippo'],
        'hippocampus': ['HIP', 'hip'],
        'Midbrain': ['Midbrain'],
        'Cerebellum': ['Cerebellum'],
    }

  df = pd.DataFrame(columns=['name', 'group', 'location', 'mouse', 'definitely alive', 'definitely dead', 'likely alive', 'likely dead'])

  for image in all_analysis:
    name = image[0]

    mouse = name.split('_')[0]  # Assuming mouseID is the first part of the name before an underscore

    group = 'other'  # Default group if no match is found
    # Iterate through the group_map to find a match
    for grp, variants in group_map.items():
      if any(variant.lower() in name.lower() for variant in variants):
        group = grp
        break  # Exit the loop once a match is found

    location = 'other'  # Default location if no match is found
    # Iterate through the location_map to find a match
    for loc, variants in location_map.items():
      if any(variant.lower() in name.lower() for variant in variants):
        location = loc
        break  # Exit the loop once a match is found


    counts = pd.Series(image[1]['alive_or_dead']).value_counts()
    likely_dead = counts.get('likely dead', 0)
    likely_alive = counts.get('likely alive', 0)
    def_dead = counts.get('definitely dead', 0)
    def_alive = counts.get('definitely alive', 0)

    df.loc[len(df)] = [name, group, location, mouse, def_alive, def_dead, likely_alive, likely_dead]


  return df

def summarize_by_mouse(analysis_df, include_likely=True, collapse_to_groups=False):
    # Work on a copy to avoid modifying the original DataFrame
    df = analysis_df.copy()

    # Extract the mouseID from the 'name' column (assumes mouseID is before the first underscore)
    df['mouseID'] = df['name'].apply(lambda s: s.split('_')[0])

    # Compute alive and dead counts based on the include_likely flag
    if include_likely:
        df['alive'] = df['definitely alive'] + df['likely alive']
        df['dead']  = df['definitely dead'] + df['likely dead']
    else:
        df['alive'] = df['definitely alive']
        df['dead']  = df['definitely dead']

    # Compute the fraction of dead cells per row
    df['fraction_dead'] = df['dead'] / (df['alive'] + df['dead'])

    # Group by mouseID, group, and location, and compute the average fraction_dead for each combination
    grouped = df.groupby(['mouseID', 'group', 'location'])['fraction_dead'].mean().reset_index()

    # Pivot the table so that each row is a mouse and each location becomes a column
    pivot_df = grouped.pivot(index='mouseID', columns='location', values='fraction_dead')

    # Get group info for each mouse (assuming one group per mouse)
    group_info = grouped.drop_duplicates(subset='mouseID').set_index('mouseID')['group']

    # Add the group info to the pivoted DataFrame
    pivot_df = pivot_df.join(group_info)

    # Add the mouseID as a column (currently it is the index)
    pivot_df['mouseID'] = pivot_df.index

    # Reorder columns so that 'mouseID' and 'group' are first, followed by the sorted location columns.
    primary_cols = ['mouseID', 'group']
    other_cols = sorted([col for col in pivot_df.columns if col not in primary_cols])
    pivot_df = pivot_df[primary_cols + other_cols]

    # If collapse_to_groups is True, group the data by 'group' and average over all mice for each location.
    if collapse_to_groups:
        # Identify the location columns (all columns except 'mouseID' and 'group')
        location_cols = [col for col in pivot_df.columns if col not in ['mouseID', 'group']]
        # Group by 'group' and take the mean for the location columns
        collapsed_df = pivot_df.groupby('group')[location_cols].mean().reset_index()
        # Reorder the columns: 'group' first, then location columns in alphabetical order
        collapsed_df = collapsed_df[['group'] + sorted(location_cols)]
        return collapsed_df

    return pivot_df