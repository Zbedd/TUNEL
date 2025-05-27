import yaml
import argparse
import sys

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

# Make package importable
module_dir = r"C:/VScode/TUNEL/"
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    
from tunel_quant import summarize, plotting, stats

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
    apply_masks = cfg.get("mask_folder") is not None
    mask_path = Path(cfg.get("mask_folder"))
    date_str = datetime.now().strftime('%Y-%m-%d')
    sex = cfg['sex']
    if sex == None:
        sex = 'MF'  # Default to 'MF' if not specified
    if cfg['include_likely']:
        likely_str = 'includeLikely'
    else:  
        likely_str = 'excludeLikely'
    
    analysis = summarize.analyze_folder(cfg['input_folder'], mask_folder=mask_path, apply_masks=apply_masks, sex = cfg['sex'], sex_path = cfg['sex_path'], method=cfg['seg_method'], conThresh=cfg['conThresh'], kSize=cfg['kSize'], magnification=cfg['magnification'])
    
    # Save the analysis data to a CSV file
    summary = summarize.summarize_analysis(analysis, cfg['l_map'])
    by_mouse = summarize.summarize_by_mouse(summary)
    by_mouse_collapsed = summarize.summarize_by_mouse(summary, collapse_to_groups=True)
    
    # Perform statistical analysis on the summary data
    anova_results = stats.anova(summary, post_hoc=False, include_likely=cfg['include_likely'])
    mixed_results = stats.mixed_logistic(summary, include_likely=cfg['include_likely'])
    
    # # Plot based on the locations mapped in location_map (stored in summary dataframe)   
    # summary_plot = plotting.plot_summary(summary, title = 'Percentage of cells alive by group (test data)', include_likely=cfg['include_likely'], plot_dots = False, plot_sample_size=True, include_location = True, include_other = False, flip_group_location=True, add_significance=True)
    
    '''WRITING FUNCTIONS'''
    # Create an output folder at the output_folder path if it doesn't exist
    folder_name = f'tunel_quant_{sex}_{likely_str}_{date_str}'
    out_folder_path = os.path.join(cfg['output_folder'], folder_name)
    os.makedirs(out_folder_path, exist_ok=True)
    
    # Write by_mouse and by_mouse_collapsed to an Excel file with two sheets
    excel_output_path = os.path.join(out_folder_path, f'summary_by_mouse_{date_str}.xlsx')
    with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        summary.to_excel(writer, sheet_name='By Image', index=False)
        by_mouse.to_excel(writer, sheet_name='By Mouse', index=False)
        by_mouse_collapsed.to_excel(writer, sheet_name='By Mouse Collapsed', index=False)
        
    # Save the anova, tukey, and agg results to an Excel file
    anova_output_path = os.path.join(out_folder_path, f'anova_results_{date_str}.xlsx')
    with pd.ExcelWriter(anova_output_path, engine='xlsxwriter') as writer:
        anova_results['anova'].to_excel(writer, sheet_name='ANOVA Results', index=False)
        if anova_results['tukey'] is not None:
            tukey_df = pd.DataFrame(anova_results['tukey'].summary())
            tukey_df.to_excel(writer, sheet_name='Tukey HSD Results', index=False)
        anova_results['agg'].to_excel(writer, sheet_name='Mouse Summary', index=False)
        
    # Save the mixed logistic results to a CSV file
    
    # Build the summary table
    level = 0.95
    z     = norm.ppf(0.5 + level / 2)          # 1.96 for 95 %
    mixed_summary = pd.DataFrame({
        "parameter"              : mixed_results.model.data.param_names,
        "log_odds_mean"          : mixed_results.fe_mean,
        "log_odds_sd"            : mixed_results.fe_sd,
        "odds_ratio"             : np.exp(mixed_results.fe_mean),
        f"lower_{int(level*100)}": np.exp(mixed_results.fe_mean - z * mixed_results.fe_sd),
        f"upper_{int(level*100)}": np.exp(mixed_results.fe_mean + z * mixed_results.fe_sd)
    })
    mixed_output_path = os.path.join(out_folder_path, f'mixed_logistic_results_{date_str}.csv')
    mixed_summary.to_csv(mixed_output_path, index=False)
        
    
    
    # Save the summary plot to a PNG file
    plot_output_path = os.path.join(out_folder_path, f'summary_plot_{date_str}.png')
    '''# summary_plot.savefig(plot_output_path, format='png')'''

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
    parser.add_argument( #input
        "--mask", "-r",
        default=cfg["mask_folder"],
        help=f"Mask folder (default: {cfg['mask_folder']})"
    )    
    parser.add_argument( #output
        "--output", "-o",
        default=cfg["output_folder"],
        help=f"Output folder (default: {cfg['output_folder']})"
    )
    parser.add_argument( #sex
        "--sex", "-s",
        default=cfg["sex"],
        help=f"Sex (default: {cfg['sex']})"
    )
    parser.add_argument( #method
        "--seg_method", "-m",
        default=cfg["seg_method"],
        help=f"Segmentation method (default: {cfg['seg_method']})"
    )
    parser.add_argument( #include_likely
        "--include_likely", "-l",
        default=cfg["include_likely"],
        help=f"Include likely (default: {cfg['include_likely']})"
    )
    

    args = parser.parse_args()
    
    cfg["input_folder"]       = args.input
    cfg["mask_folder"]        = args.mask
    cfg["output_folder"]      = args.output
    cfg["sex"]                = args.sex
    cfg["seg_method"]         = args.seg_method
    cfg["include_likely"]     = args.include_likely

    main(cfg)