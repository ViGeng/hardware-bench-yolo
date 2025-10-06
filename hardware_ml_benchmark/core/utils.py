#!/usr/bin/env python3
"""
Utilities module - contains logging setup, dependency checks and other utility functions
"""

import os
import sys
import time
import socket
import logging
import numpy as np
import torch

def setup_logging():
    """Set up logging system"""
    # Create results directory (if it doesn't exist)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate timestamped log filename
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(results_dir, f"benchmark_log_{timestamp}.log")
    
    # Configure log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized, log file: {log_filename}")
    
    return logger, log_filename

def check_dependencies():
    """Check if dependency libraries are available"""
    dependencies = {
        'ultralytics': False,
        'pynvml': False,
        'smp': False,
        'pil': False,
        'cv2': False,
        'timm': False,
        'matplotlib': False,
        'seaborn': False,
        'torchvision_detection': False,
        'tqdm': False
    }
    
    # Check ultralytics
    try:
        from ultralytics import YOLO
        dependencies['ultralytics'] = True
    except ImportError:
        pass
    
    # Check pynvml
    try:
        import pynvml
        dependencies['pynvml'] = True
    except ImportError:
        pass
    
    # Check segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp
        dependencies['smp'] = True
    except ImportError:
        pass
    
    # Check PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        dependencies['pil'] = True
    except ImportError:
        pass
    
    # Check OpenCV
    try:
        import cv2
        dependencies['cv2'] = True
    except ImportError:
        pass
    
    # Check timm
    try:
        import timm
        dependencies['timm'] = True
    except ImportError:
        pass
    
    # Check matplotlib and seaborn
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        dependencies['matplotlib'] = True
        dependencies['seaborn'] = True
    except ImportError:
        pass
    
    # Check torchvision detection
    try:
        import torchvision.models.detection as detection_models
        dependencies['torchvision_detection'] = True
    except ImportError:
        pass
    
    # Check tqdm
    try:
        from tqdm import tqdm
        dependencies['tqdm'] = True
    except ImportError:
        pass
    
    return dependencies

def print_dependency_status(dependencies):
    """Print dependency status"""
    missing_deps = []
    
    if not dependencies['ultralytics']:
        missing_deps.append("ultralytics (pip install ultralytics)")
    
    if not dependencies['pynvml']:
        missing_deps.append("nvidia-ml-py3 (pip install nvidia-ml-py3)")
    
    if not dependencies['smp']:
        missing_deps.append("segmentation-models-pytorch (pip install segmentation-models-pytorch)")
    
    if not dependencies['pil']:
        missing_deps.append("Pillow (pip install Pillow)")
    
    if not dependencies['timm']:
        missing_deps.append("timm (pip install timm)")
    
    if not dependencies['matplotlib']:
        missing_deps.append("matplotlib seaborn (pip install matplotlib seaborn)")
    
    if not dependencies['tqdm']:
        missing_deps.append("tqdm (pip install tqdm)")
    
    if missing_deps:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Missing dependencies detected")
        print("Recommend installing the following dependencies for full functionality:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()

def safe_time_value(time_value, min_value=0.001):
    """Ensure time value is reasonable, avoid abnormal data"""
    return max(time_value, min_value)

def calculate_fps(time_ms):
    """Calculate FPS from millisecond time, avoid infinity"""
    time_ms = safe_time_value(time_ms)
    return min(1000.0 / time_ms, 10000)  # Limit maximum FPS to 10000

def get_system_info():
    """Get system information"""
    return {
        'hostname': socket.gethostname(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

class GrayscaleToRGB(object):
    """Transform to convert grayscale images to RGB images"""
    def __call__(self, img):
        if img.shape[0] == 1:  # If single channel
            return img.repeat(3, 1, 1)  # Duplicate to 3 channels
        return img