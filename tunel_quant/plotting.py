"""
Visualization and Plotting Module for TUNEL Analysis

This module provides comprehensive visualization capabilities for TUNEL assay analysis,
including raw image display, segmentation overlay, statistical plotting, and 
interactive data exploration tools.

Key functionality:
- Display ND2 microscopy images with channel switching
- Overlay segmentation results with color-coded cell viability status  
- Generate statistical summary plots and visualizations
- Interactive widgets for data exploration
- Export-ready publication figures

The module supports both static matplotlib plots and interactive IPython widgets
for exploratory data analysis of cell viability measurements.
"""

import os
import numpy as np
import pandas as pd
import itertools
from scipy.stats import ttest_ind
import cv2
from IPython.display import display
import ipywidgets as widgets
import nd2reader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def plot(image, title=None, xlabel=None, ylabel=None, interpolation='nearest', colorbar=False, figsize=(8, 6)):
    """
    Display a uint8 image using matplotlib with optional customization.
    
    This is a utility function for quickly visualizing processed images with
    consistent formatting and optional colorbar display.
    
    Parameters
    ----------
    image : ndarray
        Input image data. Must be uint8 format.
    title : str, optional
        Title to display above the image.
    xlabel : str, optional  
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    interpolation : str, default='nearest'
        Interpolation method for image display. Common options: 'nearest', 'bilinear'.
    colorbar : bool, default=False
        Whether to display intensity colorbar alongside image.
    figsize : tuple, default=(8, 6)
        Figure size as (width, height) in inches.
        
    Raises
    ------
    ValueError
        If input image is not uint8 data type.
    """
    if image.dtype != np.uint8:
        raise ValueError("The provided image is not of type uint8.")
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, interpolation=interpolation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if colorbar:
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def visualize_nd2_image(folder_path, file_name, summary_df=None):
    """
    Interactive visualization of dual-channel ND2 microscopy images.
    
    This function provides an interactive interface for viewing DAPI and FITC channels
    from ND2 files, with optional annotation from analysis results. Users can switch
    between individual channels and merged false-color displays.
    
    Features:
    - Interactive dropdown for channel selection (DAPI, FITC, Merged)  
    - False-color merged display (DAPI=blue, FITC=green)
    - Optional overlay of analysis results from summary DataFrame
    - Fallback to static display if widgets unavailable
    
    Parameters
    ----------
    folder_path : str
        Directory path containing the ND2 file.
    file_name : str
        Name of the ND2 file to visualize.
    summary_df : pandas.DataFrame, optional
        Analysis results DataFrame for annotation. Should contain columns
        matching the image filename for result overlay.
        
    Notes
    -----
    - Requires nd2reader for ND2 file support
    - Uses IPython widgets for interactivity when available
    - Handles multi-timepoint and multi-z-stack files by showing first frame
    - Merged view uses false-color overlay: DAPI (blue) + FITC (green)
    """

    full_path = os.path.join(folder_path, file_name)

    # Try reading ND2 image
    try:
        with nd2reader.ND2Reader(full_path) as images:
            if 'z' in images.sizes:
                images.default_coords['z'] = 0
            if 't' in images.sizes:
                images.default_coords['t'] = 0

            if 'c' not in images.sizes or images.sizes['c'] < 2:
                raise ValueError(f"Expected at least 2 channels. Found {images.sizes.get('c', 1)}.")

            dapi = images.get_frame_2D(c=0)
            fitc = images.get_frame_2D(c=1)
    except Exception as e:
        raise IOError(f"Failed to read ND2 image file: {full_path}\n{e}")

    # Normalize channels
    dapi_norm = dapi / np.max(dapi) if np.max(dapi) > 0 else dapi
    fitc_norm = fitc / np.max(fitc) if np.max(fitc) > 0 else fitc

    # Create colorized versions
    dapi_rgb = np.stack([np.zeros_like(dapi_norm), np.zeros_like(dapi_norm), dapi_norm], axis=-1)
    fitc_rgb = np.stack([np.zeros_like(fitc_norm), fitc_norm, np.zeros_like(fitc_norm)], axis=-1)
    merged_rgb = np.stack([np.zeros_like(dapi_norm), fitc_norm, dapi_norm], axis=-1)

    # Optional annotation
    annotation = ""
    if summary_df is not None:
        match = summary_df[summary_df['file_name'] == file_name]
        if match.empty:
            raise ValueError(f"No entry in summary DataFrame for file_name: {file_name}")
        row = match.iloc[0]
        group = row.get('group', 'N/A')
        location = row.get('location', 'N/A')
        da = row.get('definitely alive', 0)
        la = row.get('likely alive', 0)
        dd = row.get('definitely dead', 0)
        ld = row.get('likely dead', 0)
        annotation = (
            f"Group: {group}\nLocation: {location}\n"
            f"Definitely Alive: {da}\nLikely Alive: {la}\n"
            f"Definitely Dead: {dd}\nLikely Dead: {ld}"
        )

    # Internal display function
    def show_channel(channel):
        fig, ax = plt.subplots(figsize=(12, 12))
        if channel == 'DAPI':
            ax.imshow(dapi_rgb)
            ax.set_title('DAPI (Blue)')
        elif channel == 'FITC':
            ax.imshow(fitc_rgb)
            ax.set_title('FITC (Green)')
        else:
            ax.imshow(merged_rgb)
            ax.set_title('Merged (FITC + DAPI)')

        if annotation:
            ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6))

        ax.axis('off')
        plt.show()

    # Interactive display
    try:
        dropdown = widgets.Dropdown(
            options=['DAPI', 'FITC', 'Merged'],
            value='Merged',
            description='Channel:'
        )
        output = widgets.interactive_output(show_channel, {'channel': dropdown})
        display(widgets.VBox([dropdown, output]))
    except:
        # Fallback to static side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(dapi_rgb)
        axs[0].set_title('DAPI (Blue)')
        axs[0].axis('off')
        axs[1].imshow(fitc_rgb)
        axs[1].set_title('FITC (Green)')
        axs[1].axis('off')
        if annotation:
            fig.suptitle(annotation, fontsize=12, y=0.95)
        plt.tight_layout()
        plt.show()

def plot_dapi_fitc(dapi, fitc, mode='side_by_side'):
    """
    Display DAPI and FITC images with customizable viewing modes.
    
    This function provides flexible visualization of dual-channel fluorescence images
    using custom colormaps that ensure proper intensity mapping from black (0) to 
    full color (255). Supports both comparative side-by-side display and merged overlay.
    
    Parameters
    ----------
    dapi : ndarray
        DAPI channel image with intensity values 0-255 (grayscale).
    fitc : ndarray
        FITC channel image with intensity values 0-255 (grayscale).
    mode : str, default='side_by_side'
        Display mode options:
        - 'side_by_side': Show channels separately with custom colormaps
        - 'overlay': Merge channels into false-color RGB image
        
    Raises
    ------
    ValueError
        If invalid mode is specified or if image shapes don't match for overlay mode.
        
    Notes
    -----
    - Uses custom linear colormaps: DAPI (black to blue), FITC (black to green)
    - Overlay mode combines channels: Red=0, Green=FITC, Blue=DAPI
    - Intensity range 0-255 is preserved with proper normalization
    """
    # Create custom colormaps for proper intensity visualization:
    # Black (intensity 0) to full color (intensity 255)
    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_cmap', [(0, 0, 0), (0, 0, 1)], N=256
    )
    green_cmap = mcolors.LinearSegmentedColormap.from_list(
        'green_cmap', [(0, 0, 0), (0, 1, 0)], N=256
    )

    if mode not in ['side_by_side', 'overlay']:
        raise ValueError("Invalid mode. Choose either 'side_by_side' or 'overlay'.")

    if mode == 'side_by_side':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Apply custom colormaps with explicit intensity range mapping
        ax1.imshow(dapi, cmap=blue_cmap, vmin=0, vmax=255)
        ax1.set_title("DAPI")
        ax1.axis('off')

        ax2.imshow(fitc, cmap=green_cmap, vmin=0, vmax=255)
        ax2.set_title("FITC")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    elif mode == 'overlay':
        if dapi.shape != fitc.shape:
            raise ValueError("For overlay mode, DAPI and FITC images must have the same shape.")

        # Normalize images to [0, 1] range for RGB composition
        dapi_norm = dapi.astype(np.float32) / 255.0
        fitc_norm = fitc.astype(np.float32) / 255.0

        # Create false-color RGB overlay:
        # Red=0 (unused), Green=FITC, Blue=DAPI
        # - Red channel: zeros
        rgb = np.zeros((*dapi.shape, 3), dtype=np.float32)
        rgb[..., 2] = dapi_norm  # Blue channel
        rgb[..., 1] = fitc_norm  # Green channel

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title("Overlay (DAPI: Blue, FITC: Green)")
        plt.axis('off')
        plt.show()

#Plots the nuclei channel color-coded by alive/dead
def color_status_labels(labels, df, color_order='RGB'):
    """
    Create a colored overlay image from a labeled image and a DataFrame containing
    nucleus IDs and their status ("alive" or "dead"). In the output image:
      - Alive nuclei are colored green.
      - Dead nuclei are colored red.
      - Background (label 0) is black.

    Parameters:
      labels (ndarray): Labeled image (each nucleus has a unique integer label).
                        May be a Cupy array.
      df (DataFrame): A pandas DataFrame with at least two columns:
                      'nucleus_id' (int) and 'alive_or_dead' (str; either "alive" or "dead").
      color_order (str): 'RGB' or 'BGR'. Determines the order of color channels in the output.

    Returns:
      colored_image (ndarray): An image where each nucleus is colored according to its status.
                                If color_order is 'RGB', alive nuclei are green and dead nuclei are red.
                                If color_order is 'BGR', the colors are swapped accordingly.
    """
    # Ensure labels is a NumPy array.
    if hasattr(labels, "get"):
        labels_np = labels.get()
    else:
        labels_np = labels.copy()

    # Create an output image with 3 channels (RGB), initialized to black.
    colored_image = np.zeros((labels_np.shape[0], labels_np.shape[1], 3), dtype=np.uint8)

    # Loop over each row in the dataframe and color the corresponding nucleus.
    for _, row in df.iterrows():
        nucleus_id = row["nucleus_id"]
        status = row["alive_or_dead"].lower()
        # Create a mask for the current nucleus.
        mask = labels_np == nucleus_id
        if status == "alive":
            # Green: in RGB, that's (0,255,0)
            colored_image[mask] = np.array([0, 255, 0], dtype=np.uint8)
        else:
            # Red: in RGB, that's (255,0,0)
            colored_image[mask] = np.array([255, 0, 0], dtype=np.uint8)

    # Convert to BGR if requested.
    if color_order.upper() == 'BGR':
        colored_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)

    return colored_image

'''Important: Likely dead is treated as alive due to biological contexts'''
def plot_summary(
    df,
    include_likely=True,
    include_location=True,
    plot_dots=True,
    plot_sample_size=True,
    add_significance=True,
    title="Summary Plot",
    include_other=True,
    flip_group_location=False
):
    """
    Plots a summary barplot of alive cell percentage, **aggregated at the mouse level**.

    At each (group, location), we collapse all images for that mouse into one
    binomial trial (sum of alive vs. dead nuclei). We then compute a mean proportion
    across mice and a 95% CI that reflects each mouse’s binomial variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'name'
          - 'group'
          - 'location'
          - 'mouse'
          - 'definitely alive'
          - 'definitely dead'
          - 'likely alive'
          - 'likely dead'

    include_likely : bool, default=True
        Whether to include “likely alive/dead” counts in the numerator/denominator.

    include_location : bool, default=True
        If True, show side‐by‐side bars for each location within each group.
        If False, collapse over location and only show group‐level bars.

    plot_dots : bool, default=True
        Whether to scatter‐plot individual mouse‐level % alive for each bar.

    plot_sample_size : bool, default=True
        Whether to annotate each bar with “n = N_mice” above it.

    add_significance : bool, default=True
        Placeholder: currently does nothing (no sig‐bars).

    title : str, default="Summary Plot"
        Title of the plot.

    include_other : bool, default=True
        If False, drop rows with group == 'other' or location == 'other'.

    flip_group_location : bool, default=False
        If True (and include_location=True), swap axes: locations on the x‐axis,
        grouped by color for each group, instead of groups on x‐axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the barplot.
    """

    # ── 0. Filter out 'other' if requested ─────────────────────────────────────
    df = df.copy()
    if not include_other:
        df = df[(df['location'] != 'other') & (df['group'] != 'other')]

    # ── 1. Determine which columns go into “alive” vs “total” ─────────────────
    if include_likely:
        alive_cols = ['definitely alive', 'likely alive', 'likely dead']
    else:
        alive_cols = ['definitely alive']

    # Regardless, “total” must include definitely dead + alive categories
    total_cols = alive_cols + ['definitely dead']

    # ── 1b. Compute per‐image alive percentage (for scatter‐dots only) ─────────
    df['alive_percent_image'] = df[alive_cols].sum(axis=1) / df[total_cols].sum(axis=1) * 100

    # ── 2. Collapse to one row per mouse (within each group & location) ────────
    #   For each mouse, sum all “alive” and all “dead” across its images.
    if include_location:
        mouse_group_cols = ['group', 'location', 'mouse']
    else:
        mouse_group_cols = ['group', 'mouse']

    df_mouse = (
        df
        .groupby(mouse_group_cols, as_index=False)
        .agg({
            'definitely alive': 'sum',
            'likely alive':     'sum',
            'definitely dead':  'sum',
            'likely dead':      'sum'
        })
    )

    # Recompute successes, failures, total, p_mouse, var_mouse
    if include_likely:
        df_mouse['successes_mouse'] = df_mouse['definitely alive'] + df_mouse['likely alive'] + df_mouse['likely dead']
        df_mouse['failures_mouse']  = df_mouse['definitely dead']  
        df_mouse['total_mouse']     = df_mouse['successes_mouse'] + df_mouse['failures_mouse']
    else: 
        df_mouse['successes_mouse'] = df_mouse['definitely alive']
        df_mouse['failures_mouse']  = df_mouse['definitely dead']
        df_mouse['total_mouse']     = df_mouse['successes_mouse'] + df_mouse['failures_mouse']

    # Prevent dividing by zero (drop any mouse with total_mouse == 0)
    df_mouse = df_mouse[df_mouse['total_mouse'] > 0].copy()

    # Mouse‐level proportion and its binomial variance
    df_mouse['p_mouse']   = df_mouse['successes_mouse'] / df_mouse['total_mouse']
    df_mouse['var_mouse'] = df_mouse['p_mouse'] * (1 - df_mouse['p_mouse']) / df_mouse['total_mouse']

    # ── 3. Compute per‐(group,location) summary at the mouse level ─────────────
    if include_location:
        group_cols = ['group', 'location']
    else:
        group_cols = ['group']

    # For each (group, location), we need:
    #   • mean_p  = average of p_mouse across all mice in that cell
    #   • se_mean = sqrt( sum(var_mouse) ) / N_mice
    #   • ci95    = 1.96 * se_mean
    summary_mouse = (
        df_mouse
        .groupby(group_cols)
        .agg(
            mean_p     = ('p_mouse',   'mean'),
            sum_var    = ('var_mouse', 'sum'),
            count_mice = ('mouse',     'nunique')
        )
        .reset_index()
    )

    # Compute SE and 95% CI on the mouse‐level mean
    summary_mouse['se_mean'] = np.sqrt(summary_mouse['sum_var']) / summary_mouse['count_mice']
    summary_mouse['ci95']    = 1.96 * summary_mouse['se_mean']

    # ── 4. Build the barplot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    max_y    = 0
    bar_centers = {}

    # ── 4a. “group + location” on x‐axis (side‐by‐side) ─────────────────────────
    if include_location and not flip_group_location:
        groups    = summary_mouse['group'].unique()
        locations = summary_mouse['location'].unique()
        x         = np.arange(len(groups))
        bar_w     = 0.8 / len(locations)
        cmap      = plt.cm.get_cmap('Set2', len(locations))

        for i, loc in enumerate(locations):
            for j, grp in enumerate(groups):
                xpos = x[j] - 0.4 + i * bar_w
                bar_centers[(grp, loc)] = xpos

                # Pull out the summary row for this (grp,loc)
                row = summary_mouse[
                    (summary_mouse['group'] == grp) &
                    (summary_mouse['location'] == loc)
                ]

                if row.empty:
                    continue

                mean_p   = row['mean_p'].iloc[0]
                ci95     = row['ci95'].iloc[0]
                n_mice   = int(row['count_mice'].iloc[0])

                # Plot bar at mean_p * 100 (to convert to percent)
                ax.bar(xpos, mean_p * 100, width=bar_w,
                       color=cmap(i), alpha=0.85, zorder=2)

                # Plot error bar ± (ci95 * 100)
                ax.errorbar(xpos, mean_p * 100, yerr=ci95 * 100,
                            fmt='none', capsize=4, color='black', zorder=3)

                # Optionally, scatter‐plot each mouse’s own %alive (image‐level 
                # values for that mouse; we’ll use the first image’s group+location 
                # to retrieve all image-level dots, but they represent the same 
                # per‐mouse proportion if multiple images exist—we’ll just plot them 
                # all for transparency)
                if plot_dots:
                    # Gather all image‐level per‐mouse proportions (alive_percent_image)
                    dots = df[
                        (df['group'] == grp) &
                        (df['location'] == loc) &
                        (df['mouse'].isin(df_mouse[df_mouse['group']==grp]['mouse']))
                    ]['alive_percent_image']

                    # Scatter‐plot them at xpos
                    if not dots.empty:
                        ax.scatter(np.full_like(dots, xpos), dots,
                                   color='black', s=15, alpha=0.6, zorder=4)
                        # Update max_y in case these dots stick out
                        max_y = max(max_y, (dots.max() if not dots.empty else 0))

                # Update max_y based on bar + CI
                top = (mean_p * 100) + (ci95 * 100)
                max_y = max(max_y, top)

                # Optionally, annotate number of mice above the bar
                if plot_sample_size:
                    ax.text(xpos, top + 2, f'n = {n_mice}',
                            ha='center', va='bottom', fontsize=9)

        # Legend for locations
        handles = [Patch(facecolor=cmap(i), label=loc)
                   for i, loc in enumerate(locations)]
        ax.legend(handles=handles, title='Location')

        if add_significance:
            print("WARNING: Significance bars in plotting not implemented yet.")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)

    # ── 4b. “Flipped” plot: location on x, bars colored by group ───────────────
    elif include_location and flip_group_location:
        locations = summary_mouse['location'].unique()
        groups    = summary_mouse['group'].unique()
        x         = np.arange(len(locations))
        bar_w     = 0.8 / len(groups)
        cmap      = plt.cm.get_cmap('Set2', len(groups))

        for i, grp in enumerate(groups):
            for j, loc in enumerate(locations):
                xpos = x[j] - 0.4 + i * bar_w
                bar_centers[(loc, grp)] = xpos

                row = summary_mouse[
                    (summary_mouse['group'] == grp) &
                    (summary_mouse['location'] == loc)
                ]
                if row.empty:
                    continue

                mean_p = row['mean_p'].iloc[0]
                ci95   = row['ci95'].iloc[0]
                n_mice = int(row['count_mice'].iloc[0])

                ax.bar(xpos, mean_p * 100, width=bar_w,
                       color=cmap(i), alpha=0.85, zorder=2)
                ax.errorbar(xpos, mean_p * 100, yerr=ci95 * 100,
                            fmt='none', capsize=4, color='black', zorder=3)

                if plot_dots:
                    dots = df[
                        (df['group'] == grp) &
                        (df['location'] == loc) &
                        (df['mouse'].isin(df_mouse[df_mouse['group']==grp]['mouse']))
                    ]['alive_percent_image']
                    if not dots.empty:
                        ax.scatter(np.full_like(dots, xpos), dots,
                                   color='black', s=15, alpha=0.6, zorder=4)
                        max_y = max(max_y, (dots.max() if not dots.empty else 0))

                top = (mean_p * 100) + (ci95 * 100)
                max_y = max(max_y, top)

                if plot_sample_size:
                    ax.text(xpos, top + 2, f'n = {n_mice}',
                            ha='center', va='bottom', fontsize=9)

        handles = [Patch(facecolor=cmap(i), label=grp) for i, grp in enumerate(groups)]
        ax.legend(handles=handles, title='Group')

        if add_significance:
            print("WARNING: Significance bars in plotting not implemented yet.")

        ax.set_xticks(x)
        ax.set_xticklabels(locations)

    # ── 4c. No location dimension: just one bar per group ──────────────────────
    else:
        groups = summary_mouse['group'].unique()
        x      = np.arange(len(groups))
        bar_w  = 0.6

        for idx, grp in enumerate(groups):
            row = summary_mouse[summary_mouse['group'] == grp]
            if row.empty:
                continue

            mean_p = row['mean_p'].iloc[0]
            ci95   = row['ci95'].iloc[0]
            n_mice = int(row['count_mice'].iloc[0])

            ax.bar(x[idx], mean_p * 100, width=bar_w,
                   color='#2ca02c' if include_likely else '#98df8a',
                   alpha=0.8, zorder=2)
            ax.errorbar(x[idx], mean_p * 100, yerr=ci95 * 100,
                        fmt='none', capsize=5, color='black', zorder=3)

            if plot_dots:
                # Plot each mouse’s %alive as a dot
                dots = df[df['group'] == grp]['alive_percent_image']
                if not dots.empty:
                    ax.scatter(np.full_like(dots, x[idx]), dots,
                               color='black', s=15, alpha=0.6, zorder=4)
                    max_y = max(max_y, (dots.max() if not dots.empty else 0))

            top = (mean_p * 100) + (ci95 * 100)
            max_y = max(max_y, top)

            if plot_sample_size:
                ax.text(x[idx], top + 2, f'n = {n_mice}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)

        if add_significance:
            print("WARNING: Significance bars in plotting not implemented yet.")

    # ── 5. Final cosmetics ─────────────────────────────────────────────────────
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('TUNEL-positive nuclei (%)')
    ax.set_title(title)
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax + 5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig