#!/usr/bin/env python3
"""
Configuration module - stores all available models, datasets and system configurations
"""

# Detection model configurations
DETECTION_MODELS = {
    '1': {'name': 'YOLOv8n', 'model': 'yolov8n.pt', 'type': 'yolo'},
    '2': {'name': 'YOLOv8s', 'model': 'yolov8s.pt', 'type': 'yolo'},
    '3': {'name': 'YOLOv8m', 'model': 'yolov8m.pt', 'type': 'yolo'},
    '4': {'name': 'Faster R-CNN ResNet50', 'model': 'fasterrcnn_resnet50_fpn', 'type': 'torchvision'},
    '5': {'name': 'Faster R-CNN MobileNet', 'model': 'fasterrcnn_mobilenet_v3_large_fpn', 'type': 'torchvision'},
    '6': {'name': 'FCOS ResNet50', 'model': 'fcos_resnet50_fpn', 'type': 'torchvision'},
}

# Classification model configurations
CLASSIFICATION_MODELS = {
    '1': {'name': 'ResNet18', 'model': 'resnet18', 'type': 'timm'},
    '2': {'name': 'ResNet50', 'model': 'resnet50', 'type': 'timm'},
    '3': {'name': 'EfficientNet-B0', 'model': 'efficientnet_b0', 'type': 'timm'},
    '4': {'name': 'EfficientNet-B3', 'model': 'efficientnet_b3', 'type': 'timm'},
    '5': {'name': 'Vision Transformer', 'model': 'vit_base_patch16_224', 'type': 'timm'},
    '6': {'name': 'MobileNet-V3', 'model': 'mobilenetv3_large_100', 'type': 'timm'},
}

# Segmentation model configurations
SEGMENTATION_MODELS = {
    '1': {'name': 'DeepLabV3+ ResNet50', 'model': 'DeepLabV3Plus', 'encoder': 'resnet50', 'type': 'smp'},
    '2': {'name': 'DeepLabV3+ EfficientNet-B0', 'model': 'DeepLabV3Plus', 'encoder': 'efficientnet-b0', 'type': 'smp'},
    '3': {'name': 'UNet ResNet34', 'model': 'Unet', 'encoder': 'resnet34', 'type': 'smp'},
    '4': {'name': 'UNet++ ResNet50', 'model': 'UnetPlusPlus', 'encoder': 'resnet50', 'type': 'smp'},
    '5': {'name': 'PSPNet ResNet50', 'model': 'PSPNet', 'encoder': 'resnet50', 'type': 'smp'},
    '6': {'name': 'FPN ResNet50', 'model': 'FPN', 'encoder': 'resnet50', 'type': 'smp'},
}

# Class label configurations
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'ostrich'
]

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# Detection color configurations
DETECTION_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]

# Segmentation color mapping
CITYSCAPES_COLOR_MAP = [
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32]     # bicycle
]

# Sample count options
SAMPLE_OPTIONS = {
    '1': {'name': 'Quick Test', 'count': 100},
    '2': {'name': 'Medium Test', 'count': 500},
    '3': {'name': 'Standard Test', 'count': 1000},
    '4': {'name': 'Large Scale Test', 'count': 5000},
    '5': {'name': 'All Samples', 'count': -1},
    '6': {'name': 'Custom Count', 'count': 'custom'}
}