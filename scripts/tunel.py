import yaml
import argparse

import os
import pandas as pd
from datetime import datetime

from tunel_quant import summarize, plotting

def main(cfg):
    import warnings
    warnings.filterwarnings(
    "ignore", 
    message="clesperanto's cupy / CUDA backend is experimental. Please use it with care.",
    category=UserWarning,
    module="pyclesperanto_prototype._tier0._cuda_backend"
    )
    valid_methods = {"otsu", "yolo"}
    if cfg["seg_method"] not in valid_methods:
        raise ValueError(f"Unknown seg_method '{cfg['seg_method']}', must be one of {valid_methods!r}")
    
    '''PIPELINE FUNCTIONS'''
        
    date_str = datetime.now().strftime('%Y-%m-%d')
    analysis = summarize.analyze_folder(cfg['input_folder'], cfg['seg_method'], cfg['conThresh'], cfg['kSize'])
    # Save the analysis data to a CSV file
    
    summary = summarize.summarize_analysis(analysis, cfg['l_map'])
    
    by_mouse = summarize.summarize_by_mouse(summary)
    by_mouse_collapsed = summarize.summarize_by_mouse(summary, collapse_to_groups=True)
    
    # Plot based on the locations mapped in location_map (stored in summary dataframe)
    summary_plot = plotting.plot_summary(summary, title = 'Percentage of cells alive by group (test data)', include_likely=True, plot_dots = False, plot_sample_size=True, include_location = True, include_other = False, flip_group_location=True, add_significance=True)
    
    '''WRITING FUNCTIONS'''
    # Write by_mouse and by_mouse_collapsed to an Excel file with two sheets
    excel_output_path = os.path.join(cfg['output_folder'], f'summary_by_mouse_{date_str}.xlsx')
    with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        summary.to_excel(writer, sheet_name='By Image', index=False)
        by_mouse.to_excel(writer, sheet_name='By Mouse', index=False)
        by_mouse_collapsed.to_excel(writer, sheet_name='By Mouse Collapsed', index=False)
    
    # Save the summary plot to a PNG file
    plot_output_path = os.path.join(cfg['output_folder'], f'summary_plot_{date_str}.png')
    summary_plot.savefig(plot_output_path, format='png')

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/default.yaml"))

    parser = argparse.ArgumentParser(
        description="Run pipeline (YAML defaults + CLI overrides)"
    )

    parser.add_argument( #input
        "--input", "-i",
        default=cfg["input_folder"],
        help=f"Input folder (default: {cfg['input_folder']})"
    )    
    parser.add_argument( #output
        "--output", "-o",
        default=cfg["output_folder"],
        help=f"Output folder (default: {cfg['output_folder']})"
    )
    parser.add_argument( #method
        "--seg_method", "-m",
        default=cfg["seg_method"],
        help=f"Segmentation method (default: {cfg['seg_method']})"
    )

    args = parser.parse_args()
    
    cfg["input_folder"]       = args.input
    cfg["output_folder"]      = args.output
    cfg["seg_method"]         = args.seg_method
    
    main(cfg)