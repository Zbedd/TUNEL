# TUNEL Analysis Pipeline

A comprehensive Python package for automated analysis of TUNEL (Terminal deoxynucleotidyl transferase dUTP nick end labeling) staining in fluorescence microscopy images. This pipeline provides end-to-end analysis from raw ND2 microscopy files to statistical results, designed specifically for cell death quantification in neuroscience research.

## Overview

The TUNEL assay is a widely used method for detecting DNA fragmentation associated with apoptotic cell death. This package automates the traditionally manual and time-intensive process of nucleus segmentation, fluorescence quantification, and statistical analysis of TUNEL-stained tissue sections.

### Key Features

- **Automated nucleus segmentation** using classical (Otsu thresholding) and deep learning (YOLO) methods
- **Dual-channel fluorescence analysis** with DAPI (nuclei) and FITC (cell death marker) channels
- **GPU acceleration** support for faster processing with CUDA-enabled PyTorch and CuPy
- **Adaptive classification** of cell viability with confidence scoring
- **Statistical analysis** including ANOVA and mixed-effects modeling
- **Batch processing** capabilities for high-throughput analysis
- **Interactive visualization** tools for quality control and data exploration

## Installation

### Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Windows, macOS, or Linux

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Zbedd/TUNEL.git
cd TUNEL
```

2. Create and activate a virtual environment:
```bash
python -m venv tunel_env
# Windows
tunel_env\Scripts\activate
# macOS/Linux
source tunel_env/bin/activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

### GPU Support (Recommended)

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA 12.4 or compatible drivers
- PyTorch with CUDA support (automatically installed with package)
- CuPy for GPU-accelerated image processing

The package will automatically detect and utilize GPU acceleration when available.

## Quick Start

### Basic Analysis

```python
from tunel_quant import summarize, stats, plotting

# Analyze all ND2 files in a folder
results = summarize.analyze_folder(
    path="path/to/nd2/images",
    method="yolo",  # or "otsu"
    apply_masks=True,
    mask_folder="path/to/masks"
)

# Generate summary statistics
summary_df = summarize.summarize_analysis(results)

# Perform statistical analysis
anova_results = stats.anova(summary_df, post_hoc=True)
print(anova_results["anova"])
```

### Command Line Interface

```bash
python scripts/tunel.py --config config/default.yaml
```

### Interactive Visualization

```python
# Visualize raw images with interactive channel switching
plotting.visualize_nd2_image(
    folder_path="path/to/images",
    file_name="sample.nd2",
    summary_df=summary_df
)

# Display segmentation results with color-coded viability
colored_labels = plotting.color_status_labels(labels, analysis_df)
plotting.plot(colored_labels, title="Cell Viability Classification")
```

## Package Structure

```
tunel_quant/
├── __init__.py          # Package initialization and configuration
├── labeling.py          # Nuclear segmentation algorithms
├── preprocessing.py     # Image enhancement and preparation
├── processing.py        # Cell viability analysis and classification
├── local_io.py          # ND2 file I/O operations
├── plotting.py          # Visualization and interactive tools
├── stats.py             # Statistical analysis functions
└── summarize.py         # Workflow orchestration and batch processing
```

## Analysis Pipeline

### 1. Image Loading and Preprocessing
- Load dual-channel ND2 microscopy files
- Extract DAPI (nuclear) and FITC (cell death) channels
- Apply noise reduction and contrast enhancement
- Normalize intensity ranges for consistent processing

### 2. Nuclear Segmentation
- **Classical method**: Otsu thresholding with morphological operations
- **Deep learning method**: YOLO-based instance segmentation
- Iterative CLAHE enhancement for challenging samples
- Size-based filtering and outlier removal

### 3. Cell Viability Classification
- Quantify FITC fluorescence within each segmented nucleus
- Calculate relative brightness using local background subtraction
- Apply statistical thresholding for confidence-based classification:
  - **Definitely dead**: High FITC signal above confidence threshold
  - **Likely dead**: Moderate FITC signal below confidence threshold
  - **Likely alive**: Low FITC signal below confidence threshold
  - **Definitely alive**: Very low FITC signal below confidence threshold

### 4. Statistical Analysis
- Aggregate results by experimental groups and individual animals
- Account for hierarchical data structure (images nested within mice)
- Perform ANOVA with post-hoc testing for group comparisons
- Generate mixed-effects models for complex experimental designs

## Configuration

Analysis parameters can be customized through YAML configuration files:

```yaml
# config/default.yaml
segmentation:
  method: "yolo"           # "otsu" or "yolo"
  splitting: true          # Watershed splitting for touching nuclei
  min_label_area: 250      # Minimum nucleus size in pixels
  
processing:
  kernel_size: 51          # Background estimation kernel
  confidence_threshold: 1.0 # Classification confidence multiplier
  
analysis:
  include_likely: true     # Include uncertain classifications
  apply_masking: false     # Use spatial region masks
```

## Input Data Requirements

### ND2 Files
- Dual-channel images with DAPI and FITC channels
- Channel metadata properly embedded in ND2 files
- Consistent imaging parameters within experiments

### Filename Convention
The pipeline extracts experimental metadata from filenames. Example:
```
881_PLX_Cre-_CA1_10x.nd2
│   │   │    │   └── Magnification
│   │   │    └────── Brain region
│   │   └─────────── Genotype
│   └─────────────── Treatment
└─────────────────── Mouse ID
```

### Optional Mask Files
- Binary TIFF images defining regions of interest
- Same dimensions as corresponding ND2 images
- Filename format: `{image_stem}_mask.tif`

## Performance Optimization

### GPU Acceleration
- Automatic detection of CUDA-compatible hardware
- GPU-accelerated image processing with CuPy
- Parallel processing for batch analysis

### Memory Management
- Efficient handling of large image datasets
- Automatic memory cleanup and garbage collection
- Configurable batch sizes for memory-constrained systems

### Processing Speed
- Typical processing time: 30-60 seconds per image (GPU)
- Batch processing scales linearly with number of cores
- YOLO segmentation: ~10x faster than classical methods

## Quality Control

### Segmentation Validation
- Interactive visualization tools for manual inspection
- Overlay segmentation results on original images
- Statistical metrics for segmentation quality assessment

### Classification Confidence
- Confidence scoring for each cell death classification
- Threshold adjustment based on experimental conditions
- Manual review capabilities for borderline cases

## Statistical Considerations

### Experimental Design
- Account for hierarchical data structure (cells within images within animals)
- Proper statistical units for hypothesis testing
- Multiple comparison corrections for post-hoc analyses

### Sample Size Planning
- Power analysis functions for experimental planning
- Effect size estimation from pilot data
- Recommendations for minimum sample sizes

## Troubleshooting

### Common Issues

**GPU not detected**: Verify CUDA installation and PyTorch GPU support
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name()}")
```

**Memory errors**: Reduce batch size or use CPU processing for large images

**Segmentation quality**: Adjust preprocessing parameters or try alternative methods

**Statistical warnings**: Check data distribution and experimental design assumptions

### Performance Monitoring
```python
# Check processing time and memory usage
import time, psutil
start_time = time.time()
# ... analysis code ...
print(f"Processing time: {time.time() - start_time:.2f} seconds")
print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")
```

## Contributing

We welcome contributions to improve the TUNEL analysis pipeline. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure documentation is updated
5. Submit a pull request

### Development Setup
```bash
pip install -e ".[dev]"  # Install development dependencies
pytest tests/            # Run test suite
black tunel_quant/       # Format code
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{tunel_analysis_pipeline,
  title={TUNEL Analysis Pipeline: Automated Cell Death Quantification},
  author={[Your Name]},
  year={2025},
  url={https://github.com/Zbedd/TUNEL}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: Comprehensive docstrings and inline comments throughout codebase
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Questions**: Contact the development team or open a discussion

## Acknowledgments

- ImageProcessingUtils package for YOLO segmentation capabilities
- OpenCV and scikit-image communities for image processing foundations
- ClearControlOpenCL (cle) for GPU-accelerated image analysis
- Statsmodels for robust statistical analysis framework

---

**Note**: This pipeline is designed for research use. Validate results with your specific experimental conditions and consult with statisticians for complex experimental designs.
