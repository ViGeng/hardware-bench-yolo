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

## Installation Guide

### 1. Clone Project
```bash
git clone <repository-url>
```

### 2. Install Basic Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn pillow tqdm psutil
```

### 3. Install Model Framework Dependencies (choose as needed)

**Image Classification Models**:
```bash
pip install timm
```

**Object Detection Models**:
```bash
pip install ultralytics  # YOLO models
# torchvision already includes detection models
```

**Semantic Segmentation Models**:
```bash
pip install segmentation-models-pytorch
```

**GPU Monitoring (Optional)**:
```bash
pip install nvidia-ml-py3
```

### 4. Verify Installation
```bash
python main.py --list-models
python main.py --list-datasets
```

## Basic Usage

### View Available Options
```bash
# List all available models
python main.py --list-models

# List all available datasets
python main.py --list-datasets

# View complete help information
python main.py --help
```

## Command Line Examples

### Image Classification Tasks

**Quick Test (CPU, 100 samples)**:
```bash
python main.py \
    --model-type classification \
    --model resnet18 \
    --dataset MNIST \
    --device cpu \
    --samples 100
```

**GPU Accelerated Test**:
```bash
python main.py \
    --model-type classification \
    --model resnet50 \
    --dataset CIFAR-10 \
    --device cuda:0 \
    --samples 500
```

**Large Scale Test (automatic device selection)**:
```bash
python main.py \
    --model-type classification \
    --model efficientnet_b0 \
    --dataset ImageNet-Sample \
    --device auto \
    --samples 1000
```

### Object Detection Tasks

**YOLO Model Test**:
```bash
python main.py \
    --model-type detection \
    --model yolov8n \
    --dataset Test-Images \
    --device cuda:0 \
    --samples 200
```

**TorchVision Detection Model**:
```bash
python main.py \
    --model-type detection \
    --model fasterrcnn-resnet50-fpn \
    --dataset COCO-Sample \
    --device auto \
    --samples 300
```

### Semantic Segmentation Tasks

**U-Net Model Test**:
```bash
python main.py \
    --model-type segmentation \
    --model unet_resnet34 \
    --dataset Synthetic-Segmentation \
    --device cuda:0 \
    --samples 150
```

**DeepLabV3+ Model Test**:
```bash
python main.py \
    --model-type segmentation \
    --model deeplabv3plus_resnet50 \
    --dataset Cityscapes \
    --device auto \
    --samples 100
```

### Advanced Options Examples

**Custom Output Directory**:
```bash
python main.py \
    --model-type classification \
    --model resnet18 \
    --dataset MNIST \
    --device auto \
    --samples 500 \
    --output-dir ./my_results
```

**Disable Chart Generation (CSV only)**:
```bash
python main.py \
    --model-type detection \
    --model yolov8s \
    --dataset Test-Images \
    --device cuda:0 \
    --samples 300 \
    --no-plots
```

**Silent Mode (reduced output)**:
```bash
python main.py \
    --model-type classification \
    --model resnet50 \
    --dataset CIFAR-10 \
    --device auto \
    --samples 200 \
    --quiet
```

**Test All Samples**:
```bash
python main.py \
    --model-type classification \
    --model resnet18 \
    --dataset MNIST \
    --device cuda:0 \
    --samples -1
```

**Adjust Monitoring Parameters**:
```bash
python main.py \
    --model-type segmentation \
    --model unet_resnet34 \
    --dataset Synthetic-Segmentation \
    --device cuda:0 \
    --samples 200 \
    --monitor-interval 0.05 \
    --disable-gpu-monitor
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