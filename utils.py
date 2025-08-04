#!/usr/bin/env python3
"""
工具模块 - 包含日志设置、依赖检查和其他实用函数
"""

import os
import sys
import time
import socket
import logging
import numpy as np
import torch

def setup_logging():
    """设置日志系统"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    log_filename = f"{hostname}_benchmark_log_{timestamp}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志记录器
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
    logger.info(f"日志系统初始化完成，日志文件: {log_filename}")
    
    return logger, log_filename

def check_dependencies():
    """检查依赖库是否可用"""
    dependencies = {
        'ultralytics': False,
        'pynvml': False,
        'smp': False,
        'pil': False,
        'cv2': False,
        'timm': False,
        'matplotlib': False,
        'seaborn': False,
        'torchvision_detection': False
    }
    
    # 检查 ultralytics
    try:
        from ultralytics import YOLO
        dependencies['ultralytics'] = True
    except ImportError:
        pass
    
    # 检查 pynvml
    try:
        import pynvml
        dependencies['pynvml'] = True
    except ImportError:
        pass
    
    # 检查 segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp
        dependencies['smp'] = True
    except ImportError:
        pass
    
    # 检查 PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        dependencies['pil'] = True
    except ImportError:
        pass
    
    # 检查 OpenCV
    try:
        import cv2
        dependencies['cv2'] = True
    except ImportError:
        pass
    
    # 检查 timm
    try:
        import timm
        dependencies['timm'] = True
    except ImportError:
        pass
    
    # 检查 matplotlib 和 seaborn
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        dependencies['matplotlib'] = True
        dependencies['seaborn'] = True
    except ImportError:
        pass
    
    # 检查 torchvision detection
    try:
        import torchvision.models.detection as detection_models
        dependencies['torchvision_detection'] = True
    except ImportError:
        pass
    
    return dependencies

def print_dependency_status(dependencies):
    """打印依赖状态"""
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
    
    if missing_deps:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到缺失依赖")
        print("建议安装以下依赖以获得完整功能:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()

def calculate_performance_rating(model_type, fps):
    """计算性能评级"""
    if model_type == 'classification':
        if fps > 100: 
            return "Excellent 🟢"
        elif fps > 50: 
            return "Good 🟡"
        elif fps > 10: 
            return "Fair 🟠"
        else: 
            return "Slow 🔴"
    elif model_type == 'detection':
        if fps > 30: 
            return "Excellent 🟢"
        elif fps > 15: 
            return "Good 🟡"
        elif fps > 5: 
            return "Fair 🟠"
        else: 
            return "Slow 🔴"
    else:  # segmentation
        if fps > 20: 
            return "Excellent 🟢"
        elif fps > 10: 
            return "Good 🟡"
        elif fps > 3: 
            return "Fair 🟠"
        else: 
            return "Slow 🔴"

def safe_time_value(time_value, min_value=0.001):
    """确保时间值合理，避免异常数据"""
    return max(time_value, min_value)

def calculate_fps(time_ms):
    """从毫秒时间计算FPS，避免无穷大"""
    time_ms = safe_time_value(time_ms)
    return min(1000.0 / time_ms, 10000)  # 限制最大FPS为10000

def get_system_info():
    """获取系统信息"""
    return {
        'hostname': socket.gethostname(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

class GrayscaleToRGB(object):
    """将灰度图像转换为RGB图像的变换"""
    def __call__(self, img):
        if img.shape[0] == 1:  # 如果是单通道
            return img.repeat(3, 1, 1)  # 复制到3个通道
        return img