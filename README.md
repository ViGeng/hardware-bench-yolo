# Deep Learning Benchmark Tool

A deep learning model performance benchmarking tool that supports image classification, object detection, semantic segmentation tasks with pretrained/customized models on multiple datasets. This tool aims to give a glance at the ML performance of your device.

## System Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA-supported GPU (optional, for GPU acceleration)
- Recommended memory: 8GB+ (depends on model size)


### Install Basic Dependencies
```bash
pip install -r requirements.txt
```

## Basic Usage

### View Available Options (if you run from source code)
```bash
# List all available models
python -m hardware_ml_benchmark.main --list-models

# List all available datasets
python -m hardware_ml_benchmark.main --list-datasets

# View complete help information
python -m hardware_ml_benchmark.main --help
```


## Possible Questions

**CUDA Related Errors**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```


**Permission Issues (Windows)**
```bash
# Run command prompt as administrator
# Or modify output directory to a directory with write permissions
python -m hardware_ml_benchmark.main --output-dir C:\Users\YourName\benchmark_results ...
```


## Command Line Examples

**Quick Test (CPU, 100 samples, classification)**:
```bash
python -m hardware_ml_benchmark.main \
    --task classification \
    --model resnet18 \
    --dataset MNIST \
    --device cpu \
    --samples 100
```

**GPU Accelerated Test(detection)**:
```bash
python -m hardware_ml_benchmark.main \
    --task detection \
    --model fasterrcnn-resnet50-fpn \
    --dataset COCO-Sample \
    --device cuda:0 \
    --samples 500
```

**Large Scale Test (automatic device selection,segmentation)**:
```bash
python -m hardware_ml_benchmark.main \
    --task segmentation \
    --model unet_resnet34 \
    --dataset Synthetic-Segmentation \
    --device auto \
    --samples 1000
```


### Advanced Options Examples

**Custom Output Directory**:
```bash
    python -m hardware_ml_benchmark.main --output-dir ./my_results
```


**Chart Generation**:
```bash
    python -m hardware_ml_benchmark.main --plot
```

**Silent Mode (reduced output)**:
```bash
    python -m hardware_ml_benchmark.main --quiet
```

**Test All Samples**:
```bash
    python -m hardware_ml_benchmark.main --samples -1
```

## Model and Dataset Support

| Task | Supported Models | Datasets | CPU Support |
|-----------|------------------|----------|-------------|
| Classification | ResNet, EfficientNet, ViT, MobileNet | MNIST, CIFAR-10, ImageNet-Sample | ✓ |
| Detection | YOLOv8, Faster R-CNN, FCOS | COCO-Sample, KITTI, Test-Images | ✓ |
| Segmentation | U-Net, DeepLabV3+, PSPNet, FPN | Cityscapes, Synthetic-Segmentation | ✓ |


##  specifying models by name, local path, or URL

# Model by name (existing functionality)
python -m hardware_ml_benchmark.main --task detection --model yolov8n --dataset Test-Images

# Local path with tilde expansion
python -m hardware_ml_benchmark.main --task detection --model ~/models/my_yolo.pt --dataset Test-Images

# Relative path
python -m hardware_ml_benchmark.main --task detection --model ./checkpoints/best.pth --dataset Test-Images

# Absolute path
python -m hardware_ml_benchmark.main --task classification --model /home/user/models/resnet.pth --dataset MNIST

# URL
python -m hardware_ml_benchmark.main --task detection --model https://example.com/models/yolo.pt --dataset Test-Images

## Limitations
-The current implementation only supports specific model formats (PyTorch .pt/.pth files, YOLO models, and framework-specific architectures).

- The current statistical reporting is limited to mean, standard deviation, min, and max values. The tool lacks advanced performance analysis including percentile distributions, confidence intervals, statistical significance testing, and performance regression detection across multiple runs.

- The implementation assumes single-device execution and lacks support for distributed inference, multi-GPU benchmarking, or batch parallelization strategies that would be essential for evaluating large-scale deployment scenarios.

## Future work
- Model Optimization Integration: Implement support for quantized models (INT8, FP16), pruned networks, and knowledge-distilled architectures to enable realistic performance comparisons between optimized and baseline models.

- Energy Efficiency Metrics: Extend monitoring to include power consumption measurements and energy-per-inference metrics, which are increasingly critical for edge deployment and sustainability considerations.

- Asynchronous Inference Profiling: Develop tools to measure pipeline parallelism, async data loading efficiency, and queue management overhead that are critical for production inference servers.

## Contribution
Past contributors: Zihan Deng
