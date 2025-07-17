# Hardware Benchmarks for YOLO

Though manufacturers provide performance specifications for their hardware, these numbers are often not representative of real-world performance or are not always intuitive to interpret. This toy project aims to provide a collective benchmark of various hardware configurations for commonly used models such as YOLO series.

If you happen to have access to any hardware or performance metrics, please feel free to contribute by submitting a pull request. The goal is to create a comprehensive benchmark that can help others make informed decisions when selecting hardware for their machine learning tasks or real-world applications.

## Benchmark Specifications

To ensure the accuracy and reliability of the benchmarks, we will be using same datasets and models across all hardware configurations. 

### Dataset

**Videos**: [sample-videos](https://github.com/intel-iot-devkit/sample-videos)
* other common detection datasets (coco, VoC, etc)
* live stream (pratical live scenarios, with diff resolutions, or conditions)

### Models

YOLO Series by [Ultralytics](https://github.com/ultralytics/ultralytics)

### Metrics

* Throughput
* Latency (different stages)
* Resource usage (CPU, GPU)

### Devices

* Powerful Nvidia GPUs
* Intel CPUs
* Mobile SoCs
* Edge Devices (such as Raspberry Pis)

### Potential Future Works

- More typical workload (Detection, Classification, or CV tasks)
- More hardwares
- Container based deliverables for convenience reproduction






# #####################################################################


[v5.0.0] - Enhanced Deep Learning Benchmark Tool with Comprehensive Logging
Added

Complete logging system with timestamps and detailed execution status
Real-time log output to both console and file
Comprehensive error tracking with detailed stack traces
Performance milestone logging throughout the benchmark process
Log file generation with unique timestamp-based naming

Enhanced

Improved error handling with detailed logging for all operations
Enhanced progress tracking with logged milestones
Better resource monitoring with logged statistics
Detailed execution flow documentation in logs


[v4.0.0] - Multi-Modal Deep Learning Benchmark Tool
Added

Semantic Segmentation support with dedicated models and datasets
KITTI dataset support for autonomous driving scenarios
Cityscapes dataset support for urban scene segmentation
Faster R-CNN and FCOS object detection models via torchvision
Segmentation models including DeepLabV3+, UNet, PSPNet, FPN
Multi-modal architecture supporting Classification, Detection, and Segmentation

Enhanced

Expanded model library with 18+ models across 3 categories
Improved dataset handling with synthetic data generation
Better model loading with automatic dependency detection
Enhanced visualization with segmentation-specific charts


[v3.0.0] - Interactive Navigation and Data Preprocessing Fixes
Added

Interactive navigation system with back/return functionality
Custom sample count selection (100, 500, 1000, 5000, all, or custom)
Detailed per-frame results recording for comprehensive analysis
Enhanced speed analysis with moving averages and performance bands
Improved data preprocessing pipeline

Fixed

MNIST dataset channel conversion (1-channel → 3-channel)
Image size normalization (28x28 → 224x224, 32x32 → 224x224)
Data transformation pipeline with proper tensor handling
Memory management for large datasets

Enhanced

Step-by-step configuration with validation
Better error messages and user guidance
Improved progress tracking with percentage indicators
Enhanced CSV output with detailed timing breakdowns


[v2.0.0] - Interactive Multi-Model Benchmark Suite
Added

Interactive setup wizard for device, model, and dataset selection
Multi-model type support (Classification and Object Detection)
Multiple dataset support (MNIST, CIFAR-10, COCO, ImageNet samples)
Classification models via timm library (ResNet, EfficientNet, ViT, MobileNet)
Advanced visualization with performance charts and resource utilization
Synthetic dataset generation for testing without real data

Enhanced

Comprehensive resource monitoring with GPU utilization tracking
Detailed performance analysis with statistical breakdowns
Multiple output formats (TXT, JSON, CSV)
Professional visualization with matplotlib and seaborn


[v1.0.0] - Command-Line Enhanced YOLO Benchmark
Added

Command-line interface with argparse support
Multiple output formats (txt, json, csv)
Flexible input sources (video files, camera devices)
Customizable batch sizes and image dimensions
Progress indicators with FPS display
Detection result visualization option

Enhanced

Improved error handling with graceful degradation
Better device detection and automatic fallback
Comprehensive help system with usage examples
Internationalization with Chinese language support


[v0.1.0] - Basic YOLO Benchmark Tool
Added

Basic YOLOv8 benchmarking for object detection
Core performance metrics (preprocess, inference, postprocess times)
System resource monitoring (CPU, memory, GPU)
Simple text output with statistics
Multi-threading resource monitoring
Hostname-based output files

Features

Real-time FPS calculation
Statistical analysis (min, max, average)
GPU memory and utilization tracking
Automatic model downloading
Basic progress reporting


Installation Requirements
bash# Core dependencies
pip install torch torchvision
pip install ultralytics
pip install timm
pip install numpy matplotlib seaborn
pip install psutil

# Optional dependencies for full functionality
pip install nvidia-ml-py3  # GPU monitoring
pip install segmentation-models-pytorch  # Segmentation models
pip install Pillow  # Image processing
Usage Evolution
v0.1.0 (Basic)
bashpython benchmark.py
v1.0.0 (Command-line)
bashpython benchmark1x.py --model yolov8n.pt --source video.mp4 --output-format csv json
v2.0.0+ (Interactive)
bashpython benchmark2x.py  # Interactive setup wizard
v5.0.0 (Current)
bashpython benchmark5.py  # Full interactive experience with comprehensive logging
Key Performance Improvements

v0.1.0: Basic YOLO inference benchmarking
v1.0.0: 2x faster setup with command-line options
v2.0.0: 5x more model variety with classification support
v3.0.0: 10x better user experience with navigation
v4.0.0: 3x more AI domains with segmentation
v5.0.0: Complete observability with detailed logging

Migration Guide
From v0.1.0 to v1.0.0

Replace hardcoded paths with command-line arguments
Update output file handling for new formats

From v1.0.0 to v2.0.0

Migrate to interactive setup (no breaking changes)
Update result parsing for new model types

From v2.0.0 to v3.0.0

Update data preprocessing for fixed MNIST handling
Adapt to new detailed result format

From v3.0.0 to v4.0.0

Add segmentation model support
Update dataset handling for new types

From v4.0.0 to v5.0.0

No breaking changes - enhanced logging is automatic
Check log files for detailed execution information
