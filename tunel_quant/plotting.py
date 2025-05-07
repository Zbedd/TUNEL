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

#Plots a uint8 image. If colorbar is true, a colorbar is included.
def plot(image, title=None, xlabel=None, ylabel=None, interpolation='nearest', colorbar=False, figsize=(8, 6)):
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


#Visualize_nd2_image and plot_dapi_fitc are two ways to plot raw images
def visualize_nd2_image(folder_path, file_name, summary_df=None):
    """
    Visualizes a 2-channel ND2 image (DAPI and FITC) from the given path and file name.

    - Uses nd2reader to load non-TIFF ND2 files
    - Shows interactive dropdown to toggle between DAPI, FITC, and Merged (colorized)
    - Falls back to static side-by-side display if widgets are unavailable
    - Optionally annotates image using matching row in summary DataFrame
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
    Plots DAPI and FITC images either side by side or overlaid.
    For side-by-side mode, custom colormaps are used so that 0 maps to black
    and 255 maps to the full color (blue for DAPI, green for FITC).

    Parameters:
      dapi (ndarray): Grayscale image for the DAPI channel (values 0-255).
      fitc (ndarray): Grayscale image for the FITC channel (values 0-255).
      mode (str): 'side_by_side' or 'overlay'. Defaults to 'side_by_side'.

    Raises:
      ValueError: If an invalid mode is provided or if image shapes don't match in overlay mode.
    """
    # Create custom colormaps:
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

        # Use the custom colormaps. We set vmin=0 and vmax=255 to ensure proper mapping.
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

        # Convert images to float and normalize to [0, 1]
        dapi_norm = dapi.astype(np.float32) / 255.0
        fitc_norm = fitc.astype(np.float32) / 255.0

        # Create an RGB image with:
        # - Blue channel: normalized DAPI
        # - Green channel: normalized FITC
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

def plot_summary(df, include_likely=True, include_location=True, plot_dots=True,
                 plot_sample_size=True, add_significance=True, title="Summary Plot",
                 include_other=True, flip_group_location=False):
    """
    Plots a summary barplot of alive cell percentage.

    By default, the x‑axis shows 'group' (with sub‑bars for each 'location' when
    include_location is True).  If flip_group_location is True (and include_location
    is True), the x‑axis shows 'location' with sub‑bars for each group.

    Optional features
    -----------------
      • include_likely      – include “likely alive/dead” counts in percentages
      • plot_dots           – scatter raw sample points on each bar
      • plot_sample_size    – print “n = …” above each bar
      • add_significance    – Welch‑t tests + significance stars
      • include_other       – drop rows whose group/location == 'other'
    """

    if flip_group_location and not include_location:
        raise ValueError("flip_group_location can only be True if include_location is True")

    df = df.copy()

    # ── 0. pre‑filter 'other' rows ─────────────────────────────────────────────
    if not include_other:
        df = df[(df['location'] != 'other') & (df['group'] != 'other')]

    # ── 1. compute alive % at the row level ───────────────────────────────────
    if include_likely:
        alive_cols = ['definitely alive', 'likely alive']
        total_cols = alive_cols + ['definitely dead', 'likely dead']
    else:
        alive_cols = ['definitely alive']
        total_cols = alive_cols + ['definitely dead']

    df['alive_percent'] = df[alive_cols].sum(axis=1) / df[total_cols].sum(axis=1) * 100

    # group either by (group, location) or just group
    group_cols = ['group', 'location'] if include_location else ['group']
    summary = (df.groupby(group_cols)['alive_percent']
                 .agg(['mean', 'std', 'count'])
                 .reset_index())
    summary['sem']  = summary['std'] / np.sqrt(summary['count'])
    summary['ci95'] = summary['sem'] * 1.96

    # ── 2. plotting scaffold ──────────────────────────────────────────────────
    fig, ax   = plt.subplots(figsize=(12, 6))
    max_y     = 0           # track max so we can place sig bars
    bar_centers = {}        # map (group,loc) → x position (or reverse if flipped)

    # ── 3a. group + location both shown (default orientation) ─────────────────
    if include_location and not flip_group_location:
        groups     = summary['group'].unique()
        locations  = summary['location'].unique()
        x          = np.arange(len(groups))
        bar_width  = 0.8 / len(locations)
        color_map  = plt.cm.get_cmap('Set2', len(locations))

        for i, loc in enumerate(locations):
            for j, grp in enumerate(groups):
                xpos = x[j] - 0.4 + i * bar_width
                bar_centers[(grp, loc)] = xpos
                row = summary[(summary['group'] == grp) &
                              (summary['location'] == loc)]
                if row.empty:
                    continue

                mean, ci, n = row[['mean', 'ci95', 'count']].values[0]
                ax.bar(xpos, mean, width=bar_width,
                       color=color_map(i), alpha=0.85, zorder=2)
                ax.errorbar(xpos, mean, yerr=ci, fmt='none',
                            capsize=4, color='black', zorder=3)

                dots = df[(df['group'] == grp) &
                          (df['location'] == loc)]['alive_percent']
                top  = max(mean + ci, dots.max() if not dots.empty else 0)
                max_y = max(max_y, top)

                if plot_sample_size:
                    ax.text(xpos, top + 2, f'n = {n}', ha='center', va='bottom', fontsize=9)
                if plot_dots:
                    ax.scatter(np.full_like(dots, xpos), dots,
                               color='black', s=15, alpha=0.6, zorder=4)

        # explicit legend  (all locations that actually appeared)
        handles = [Patch(facecolor=color_map(i), label=loc)
                   for i, loc in enumerate(locations)]
        ax.legend(handles=handles, title='Location')

        # significance between groups within each location
        if add_significance:
            line_spacing = 8
            sig_level    = 0
            for loc in locations:
                for g1, g2 in itertools.combinations(groups, 2):
                    s1 = df[(df['group'] == g1) & (df['location'] == loc)]['alive_percent']
                    s2 = df[(df['group'] == g2) & (df['location'] == loc)]['alive_percent']
                    if len(s1) > 1 and len(s2) > 1:
                        _, p = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
                        if p < 0.05:
                            x1, x2 = bar_centers[(g1, loc)], bar_centers[(g2, loc)]
                            y      = max_y + (sig_level + 1) * line_spacing
                            stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                            ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y],
                                    color='black', linewidth=1, zorder=5)
                            ax.text((x1 + x2)/2, y + 1.5, stars,
                                    ha='center', va='bottom', fontsize=14, zorder=5)
                            sig_level += 1

        ax.set_xticks(x)
        ax.set_xticklabels(groups)

    # ── 3b. flipped orientation (location on x‑axis, colours = group) ─────────
    elif include_location and flip_group_location:
        locations  = summary['location'].unique()
        groups     = summary['group'].unique()
        x          = np.arange(len(locations))
        bar_width  = 0.8 / len(groups)
        color_map  = plt.cm.get_cmap('Set2', len(groups))

        for i, grp in enumerate(groups):
            for j, loc in enumerate(locations):
                xpos = x[j] - 0.4 + i * bar_width
                bar_centers[(loc, grp)] = xpos
                row = summary[(summary['group'] == grp) &
                              (summary['location'] == loc)]
                if row.empty:
                    continue

                mean, ci, n = row[['mean', 'ci95', 'count']].values[0]
                ax.bar(xpos, mean, width=bar_width,
                       color=color_map(i), alpha=0.85, zorder=2)
                ax.errorbar(xpos, mean, yerr=ci, fmt='none',
                            capsize=4, color='black', zorder=3)

                dots = df[(df['group'] == grp) &
                          (df['location'] == loc)]['alive_percent']
                top  = max(mean + ci, dots.max() if not dots.empty else 0)
                max_y = max(max_y, top)

                if plot_sample_size:
                    ax.text(xpos, top + 2, f'n = {n}', ha='center', va='bottom', fontsize=9)
                if plot_dots:
                    ax.scatter(np.full_like(dots, xpos), dots,
                               color='black', s=15, alpha=0.6, zorder=4)

        # explicit legend (all groups that appeared)
        handles = [Patch(facecolor=color_map(i), label=grp)
                   for i, grp in enumerate(groups)]
        ax.legend(handles=handles, title='Group')

        # significance between groups within each location
        if add_significance:
            line_spacing = 8
            sig_level = {loc: 0 for loc in locations}
            for loc in locations:
                for g1, g2 in itertools.combinations(groups, 2):
                    s1 = df[(df['group'] == g1) & (df['location'] == loc)]['alive_percent']
                    s2 = df[(df['group'] == g2) & (df['location'] == loc)]['alive_percent']
                    if len(s1) > 1 and len(s2) > 1:
                        _, p = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
                        if p < 0.05:
                            x1, x2 = bar_centers[(loc, g1)], bar_centers[(loc, g2)]
                            y      = max_y + (sig_level[loc] + 1) * line_spacing
                            stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                            ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y],
                                    color='black', linewidth=1, zorder=5)
                            ax.text((x1 + x2)/2, y + 1.5, stars,
                                    ha='center', va='bottom', fontsize=14, zorder=5)
                            sig_level[loc] += 1

        ax.set_xticks(x)
        ax.set_xticklabels(locations)

    # ── 3c. only group on the x‑axis (no locations) ───────────────────────────
    else:
        x = np.arange(len(summary))
        bar_centers = dict(zip(summary['group'], x))
        for xi, grp in zip(x, summary['group']):
            mean, ci, n = summary.loc[summary['group'] == grp,
                                      ['mean', 'ci95', 'count']].values[0]
            ax.bar(xi, mean, width=0.6,
                   color='#2ca02c' if include_likely else '#98df8a', alpha=0.8, zorder=2)
            ax.errorbar(xi, mean, yerr=ci, fmt='none',
                        capsize=5, color='black', zorder=3)

            dots = df[df['group'] == grp]['alive_percent']
            top  = max(mean + ci, dots.max() if not dots.empty else 0)
            max_y = max(max_y, top)

            if plot_sample_size:
                ax.text(xi, top + 2, f'n = {n}', ha='center', va='bottom', fontsize=9)
            if plot_dots:
                ax.scatter(np.full_like(dots, xi), dots,
                           color='black', s=15, alpha=0.6, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(summary['group'])

        # significance between treatment groups
        if add_significance:
            line_spacing = 8
            sig_level    = 0
            for g1, g2 in itertools.combinations(summary['group'], 2):
                s1 = df[df['group'] == g1]['alive_percent']
                s2 = df[df['group'] == g2]['alive_percent']
                if len(s1) > 1 and len(s2) > 1:
                    _, p = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
                    if p < 0.05:
                        x1, x2 = bar_centers[g1], bar_centers[g2]
                        y      = max_y + (sig_level + 1) * line_spacing
                        stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                        ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y],
                                color='black', linewidth=1, zorder=5)
                        ax.text((x1 + x2)/2, y + 1.5, stars,
                                ha='center', va='bottom', fontsize=14, zorder=5)
                        sig_level += 1

    # ── 4. cosmetics ──────────────────────────────────────────────────────────
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Alive Cells (%)')
    ax.set_title(title)

    ax.relim(); ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax + 5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig