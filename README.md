# Deep Learning Benchmark Tool

A deep learning model performance benchmarking tool that supports image classification, object detection, and semantic segmentation tasks.

## System Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA-supported GPU (optional, for GPU acceleration)
- Recommended memory: 8GB+ (depends on model size)

## Installation Guide

### 1. Clone Project
```bash
git clone https://github.com/Zihan-D/hardware-bench-yolo.git
```

### 2. Install Basic Dependencies
```bash
pip install -r requirements.txt
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

**Quick Test (CPU, 100 samples, classification)**:
```bash
python main.py \
    --model-type classification \
    --model resnet18 \
    --dataset MNIST \
    --device cpu \
    --samples 100
```

**GPU Accelerated Test(detection)**:
```bash
python main.py \
    --model-type detection \
    --model fasterrcnn-resnet50-fpn \
    --dataset COCO-Sample \
    --device cuda:0 \
    --samples 500
```

**Large Scale Test (automatic device selection,segmentation)**:
```bash
python main.py \
    --model-type segmentation \
    --model unet_resnet34 \
    --dataset Synthetic-Segmentation \
    --device auto \
    --samples 1000
```


### Advanced Options Examples

**Custom Output Directory**:
```bash
    --output-dir ./my_results
```


**Disable Chart Generation (CSV only)**:
```bash
    --no-plots
```

**Silent Mode (reduced output)**:
```bash
    --quiet
```

**Test All Samples**:
```bash
    --samples -1
```

## Model and Dataset Support

| Task Type | Supported Models | Datasets | CPU Support |
|-----------|------------------|----------|-------------|
| Image Classification | ResNet, EfficientNet, ViT, MobileNet | MNIST, CIFAR-10, ImageNet-Sample | ✓ |
| Object Detection | YOLOv8, Faster R-CNN, FCOS | COCO-Sample, KITTI, Test-Images | ✓ |
| Semantic Segmentation | U-Net, DeepLabV3+, PSPNet, FPN | Cityscapes, Synthetic-Segmentation | ✓ |


##  specifying models by name, local path, or URL

# Model by name (existing functionality)
python main.py --model-type detection --model yolov8n --dataset Test-Images

# Local path with tilde expansion
python main.py --model-type detection --model ~/models/my_yolo.pt --dataset Test-Images

# Relative path
python main.py --model-type detection --model ./checkpoints/best.pth --dataset Test-Images

# Absolute path
python main.py --model-type classification --model /home/user/models/resnet.pth --dataset MNIST

# URL
python main.py --model-type detection --model https://example.com/models/yolo.pt --dataset Test-Images



### Q&A

**1. CUDA Related Errors**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If not available, use CPU
python main.py --device cpu ...
```

**2. Model Download Failures**
```bash
# Check network connection, models will be automatically downloaded from the internet
# YOLO models will be downloaded to ~/.cache/ultralytics/
# TIMM models will be downloaded to ~/.cache/torch/
```

**3. Out of Memory**
```bash
# Reduce the number of samples
python main.py --samples 50 ...

# Use smaller models
python main.py --model resnet18 ...  # 而不是 resnet50
```

**4. Missing Dependencies**
```bash
# Install corresponding libraries based on error messages
pip install timm  # for classification models
pip install ultralytics  # for YOLO detection
pip install segmentation-models-pytorch  # for segmentation models
```

**5. Permission Issues (Windows)**
```bash
# Run command prompt as administrator
# Or modify output directory to a directory with write permissions
python main.py --output-dir C:\Users\YourName\benchmark_results ...
```