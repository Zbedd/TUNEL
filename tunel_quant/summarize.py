import pandas as pd
import time
from . import labeling, local_io, processing

def analyze_folder(path, method = 'otsu', conThresh = 0.8, kSize = 31):
    """
    Analyzes all ND2 images in a given folder.

    For each ND2 image in the folder, the function:
      1. Loads the images using pull_nd2_images.
      2. Unpacks each image into its components: a name, a DAPI channel image, and a FITC channel image.
      3. Performs nuclear segmentation on the DAPI image using label_nuclei, with outlier removal enabled.
      4. Analyzes the nuclei in the FITC image using analyze_nuclei to extract brightness and viability (alive/dead) information.
      5. Aggregates the results into a list pairing the image name with its analysis data.

    Parameters:
      path (str): The path to the folder containing ND2 image files.
      method (str): The method for nuclear labeling. Can be 'otsu' or 'yolo'.
      conThresh (float): Weighting for certainty of alive or dead. Must be >= 1.

    Returns:
      list: A list of entries [name, analysis], where:
            - name is the image identifier.
            - analysis is the output DataFrame (or similar structure) from analyze_nuclei.
    """
    all_analysis = []  # List to store analysis results for each image.

    # Load all ND2 images from the folder.
    images = local_io.pull_nd2_images(path)

    image_count = 0
    timestamp = time.time()
    print(f"Analyzing {len(images)} images...")

    for image in images:
        # Unpack the image components:
        # image[0] -> name, image[1] -> DAPI channel image, image[2] -> FITC channel image.
        name = image[0]
        dapi = image[1]
        fitc = image[2]

        # Perform nuclear labeling on the DAPI image, with both large and small outliers removed.
        nucLabels, nucLabelStats = labeling.label_nuclei(dapi, remove_large_outliers=True, remove_small_outliers=True, method=method)

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
  form of a dataframe with columns ['group','location', 'name', 'definitely alive', 'definitely dead', 'likely alive', 'likely dead']
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

  df = pd.DataFrame(columns=['group', 'location', 'name', 'definitely alive', 'definitely dead', 'likely alive', 'likely dead'])

  for image in all_analysis:
    name = image[0]

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

    df.loc[len(df)] = [group, location, name, def_alive, def_dead, likely_alive, likely_dead]


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