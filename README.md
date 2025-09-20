# Deep Learning Benchmark Tool

A deep learning model performance benchmarking tool that supports image classification, object detection, and semantic segmentation tasks.

## Features

- **Multi-task Support**: Image classification, object detection, semantic segmentation
- **Multiple Model Frameworks**: TIMM, YOLO, TorchVision, Segmentation Models PyTorch
- **Flexible Computing Device Selection**: CPU, GPU or automatic selection
- **Real-time Resource Monitoring**: CPU, memory, GPU usage monitoring
- **Detailed Performance Analysis**: Detailed time breakdown for each sample
- **Visual Reports**: Automatic generation of performance charts and statistical reports
- **Multiple Output Formats**: CSV data files + PNG visualization charts
- **Command Line Interface**: Easy-to-use command line tool

## System Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA-supported GPU (optional, for GPU acceleration)
- Recommended memory: 8GB+ (depends on model size)


## Basic Usage

### View Available Options
```bash
# List all available models
hardware-benchmark --list-models

# List all available datasets
hardware-benchmark --list-datasets

# View complete help information
hardware-benchmark --help

# Arguments example
hardware-benchmark [-h] [--device {cpu,cuda:0,auto}] [--model-type {classification,detection,segmentation}] [--model MODEL] [--dataset DATASET] [--samples SAMPLES] [--batch-size BATCH_SIZE] [--output-dir OUTPUT_DIR] [--no-plots] [--quiet] [--list-models] [--list-datasets] [--disable-gpu-monitor] [--monitor-interval MONITOR_INTERVAL] [--monitor-samples MONITOR_SAMPLES]
```


## Output File Description

After completion, the program generates the following files in the specified output directory:

### 1. CSV Data Files

**Detailed Time Data File** (`{model_type}_detailed_{timestamp}.csv`):
- Contains detailed time data for each test sample
- Columns include: Sample_ID, Preprocessing_Time_ms, Inference_Time_ms, Postprocessing_Time_ms, Rendering_Time_ms, Total_Time_ms
- Used for in-depth analysis of individual sample performance

**System Information and Performance Summary File** (`{model_type}_summary_{timestamp}.csv`):
- Contains test environment information, overall performance metrics, and resource usage statistics
- System information: model, dataset, device, PyTorch version, etc.
- Performance metrics: throughput, average processing time, total samples, etc.
- Resource usage: CPU, memory, GPU usage statistics

### 2. Visualization Chart Files

**Detailed Speed Analysis Chart** (`{model_type}_speed_analysis_{timestamp}.png`):
- Upper half: FPS trend for each sample (raw data + moving average)
- Lower half: Time breakdown stacked chart for each sample
- Used for analyzing performance stability and bottleneck identification

**Comprehensive Performance Summary Chart** (`{model_type}_summary_{timestamp}.png`):
- 4 subplots: time breakdown pie chart, resource utilization bar chart, time distribution histogram, system information
- Used for overall performance evaluation and reporting

### 3. Log Files

**Detailed Log File** (`benchmark_log_{timestamp}.log`):
- Contains complete test process records
- Test configuration, loading process, error information, etc.
- Used for troubleshooting and test reproduction

## Device Selection Guide

### --device Parameter Options:

- **`cpu`**: Force use of CPU for computation
- **`cuda:0`**: Force use of GPU for computation (requires CUDA support)
- **`auto`**: Automatically select the best device (recommended)
  - Automatically uses GPU when available
  - Automatically uses CPU when GPU is not available

### Device Performance Comparison:

| Device Type | Speed | Memory Requirements | Use Cases |
|-------------|-------|-------------------|-----------|
| CPU | Slower | Lower | Small models, compatibility testing |
| GPU | Fast | Higher | Large models, production environments |

## Model and Dataset Support

### Supported Model Types

| Task Type | Supported Models | Dependencies |
|-----------|------------------|--------------|
| Image Classification | ResNet, EfficientNet, ViT, MobileNet | timm |
| Object Detection | YOLOv8, Faster R-CNN, FCOS | ultralytics, torchvision |
| Semantic Segmentation | U-Net, DeepLabV3+, PSPNet, FPN | segmentation-models-pytorch |

### Supported Datasets

| Task Type | Dataset | Description |
|-----------|---------|-------------|
| Image Classification | MNIST, CIFAR-10, ImageNet-Sample | Real datasets + synthetic data |
| Object Detection | COCO-Sample, KITTI, Test-Images | Synthetic detection data |
| Semantic Segmentation | Cityscapes, Synthetic-Segmentation | Urban scenes + synthetic data |


### Update log

`0.1.2` fix dependencies, fix code, modify prompt UI

`0.1.1` fix bug, fix path in code, rewrite README

`0.1.0` first release