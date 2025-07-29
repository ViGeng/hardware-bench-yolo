#!/usr/bin/env python3
"""
支持多种模型类型和数据集的交互式基准测试

新增功能：
1. KITTI数据集支持
2. Faster R-CNN和FCOS目标检测模型
3. 语义分割模式(Segmentation)，包含DeepLab、PSPNet、UNet等模型
4. 完整的日志记录系统，包含时间戳和详细的执行状态

修复了MNIST数据集的通道数不匹配问题，并添加了自定义样本数量功能

主要功能：
1. 交互式设备选择 (CPU/GPU) - 支持返回上一级
2. 模型类型选择 (Detection/Classification/Segmentation) - 支持返回上一级
3. 数据集选择 (MNIST/CIFAR-10/COCO/ImageNet/KITTI/Cityscapes) - 支持返回上一级
4. 自定义样本数量选择 - 从100到全部数据集
5. 详细性能统计和CSV输出 - 按帧/图像输出详细时间信息
6. 结果可视化
7. 完整的日志记录系统

时间测量阶段：
- Preprocessing: 数据预处理时间
- Inference: 模型推理时间  
- Postprocessing: 后处理时间
- Rendering: 渲染时间
- Total: 总时间

修复内容：
- 修复MNIST数据集的1通道到3通道转换问题
- 添加了适当的图像尺寸调整
- 改进了数据预处理流程
- 添加了完全可自定义的样本数量功能
- 新增KITTI数据集支持
- 新增Faster R-CNN、FCOS检测模型
- 新增语义分割模式和相关模型
- 新增完整的日志记录功能

Platform: Ubuntu 22.04 + NVIDIA GPU support
"""

import argparse
import json
import csv
import os
import sys
import time
import socket
import threading
import logging
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import seaborn as sns

# Deep Learning Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm

# System Monitoring
import psutil
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")

# Object Detection (if available)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Torchvision models for detection
try:
    import torchvision.models.detection as detection_models
    TORCHVISION_DETECTION_AVAILABLE = True
except ImportError:
    TORCHVISION_DETECTION_AVAILABLE = False
    print("Warning: torchvision detection models not available")

# Segmentation models
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")

# Additional libraries for segmentation and rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

# OpenCV for rendering (optional)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")


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


class GrayscaleToRGB(object):
    """将灰度图像转换为RGB图像的变换"""
    def __call__(self, img):
        if img.shape[0] == 1:  # 如果是单通道
            return img.repeat(3, 1, 1)  # 复制到3个通道
        return img


class KITTIDataset(Dataset):
    """KITTI数据集类"""
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # 获取logger
        self.logger = logging.getLogger(__name__)
        
        # 图像路径
        self.image_dir = self.root_dir / split / 'image_2'
        
        # 获取所有图像文件
        if self.image_dir.exists():
            self.image_files = sorted(list(self.image_dir.glob('*.png')))
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KITTI数据集找到 {len(self.image_files)} 个图像文件")
        else:
            # 如果没有KITTI数据，创建合成数据
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KITTI数据路径不存在: {self.image_dir}")
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 将使用合成数据进行测试")
            self.image_files = [f"synthetic_kitti_{i:06d}.png" for i in range(1000)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # 创建合成KITTI样式图像 (375x1242是KITTI的典型尺寸)
            # 使用numpy数组而不是tensor，避免transform错误
            img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        else:
            # 加载真实图像
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)  # 转换为numpy数组以保持一致性
            else:
                # 如果没有PIL，创建numpy数组
                img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, 0  # 返回图像和虚拟标签


class CityscapesDataset(Dataset):
    """Cityscapes数据集类（用于分割）"""
    def __init__(self, root_dir, split='val', transform=None, target_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 获取logger
        self.logger = logging.getLogger(__name__)
        
        # 图像和标签路径
        self.image_dir = self.root_dir / 'leftImg8bit' / split
        self.label_dir = self.root_dir / 'gtFine' / split
        
        # 获取图像文件
        if self.image_dir.exists():
            self.image_files = []
            for city_dir in self.image_dir.iterdir():
                if city_dir.is_dir():
                    self.image_files.extend(list(city_dir.glob('*_leftImg8bit.png')))
            self.image_files = sorted(self.image_files)
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cityscapes数据集找到 {len(self.image_files)} 个图像文件")
        else:
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cityscapes数据路径不存在: {self.image_dir}")
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 将使用合成数据进行测试")
            self.image_files = [f"synthetic_cityscapes_{i:06d}.png" for i in range(500)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # 创建合成Cityscapes样式图像 (512x1024，减小尺寸以加快处理)
            # 直接创建numpy数组，避免tensor转换问题
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)  # 19个类别
        else:
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)  # 转换为numpy数组
                # 尝试找到对应的标签文件
                label_path = str(img_path).replace('leftImg8bit', 'gtFine_labelIds').replace('leftImg8bit.png', 'gtFine_labelIds.png')
                if os.path.exists(label_path):
                    mask = Image.open(label_path)
                    mask = np.array(mask)
                else:
                    mask = np.random.randint(0, 19, (img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                # 创建numpy数组而不是tensor
                img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
                mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return img, mask


class RenderingEngine:
    """渲染引擎 - 负责绘制各种模型的输出结果"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # COCO类别名称（用于检测）
        self.coco_classes = [
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
        
        # ImageNet前10个类别（用于分类）
        self.imagenet_classes = [
            'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
            'electric ray', 'stingray', 'cock', 'hen', 'ostrich'
        ]
        
        # Cityscapes类别（用于分割）
        self.cityscapes_classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        # 用于检测的颜色
        self.detection_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
    
    def render_classification_result(self, image, predictions, top_k=3):
        """渲染分类结果"""
        try:
            # 模拟分类结果渲染
            if isinstance(image, torch.Tensor):
                # 将tensor转换为numpy array
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.shape[0] == 3:  # CHW -> HWC
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
            
            # 确保图像在0-255范围内
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # 模拟获取top-k预测结果
            if isinstance(predictions, torch.Tensor):
                probs = torch.nn.functional.softmax(predictions, dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs.cpu().numpy().flatten()
                top_indices = top_indices.cpu().numpy().flatten()
            else:
                # 如果是numpy array，创建模拟数据
                top_indices = np.random.choice(len(self.imagenet_classes), top_k, replace=False)
                top_probs = np.random.random(top_k)
                top_probs = top_probs / top_probs.sum()  # 归一化
            
            # 使用PIL进行文本渲染（模拟）
            if PIL_AVAILABLE:
                height, width = image.shape[:2]
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # 绘制分类结果
                y_offset = 10
                for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                    class_name = self.imagenet_classes[idx % len(self.imagenet_classes)]
                    text = f"{class_name}: {prob:.3f}"
                    draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
                    y_offset += 25
                
                # 转换回numpy数组
                rendered_image = np.array(pil_image)
            else:
                # 如果没有PIL，简单返回原图像
                rendered_image = image
            
            return rendered_image
            
        except Exception as e:
            self.logger.warning(f"分类结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((224, 224, 3), dtype=np.uint8)
    
    def render_detection_result(self, image, predictions, conf_threshold=0.5):
        """渲染检测结果（绘制边界框和置信度）"""
        try:
            # 处理输入图像
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.shape[0] == 3:  # CHW -> HWC
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
            
            # 确保图像在0-255范围内
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            height, width = image.shape[:2]
            
            # 模拟检测结果
            if hasattr(predictions, '__len__') and len(predictions) > 0:
                # 处理YOLO结果
                if hasattr(predictions[0], 'boxes') and hasattr(predictions[0], 'conf'):
                    pred = predictions[0]
                    if len(pred.boxes) > 0:
                        boxes = pred.boxes.xyxy.cpu().numpy()
                        confs = pred.conf.cpu().numpy()
                        classes = pred.cls.cpu().numpy() if hasattr(pred, 'cls') else np.zeros(len(boxes))
                    else:
                        boxes, confs, classes = [], [], []
                # 处理torchvision检测结果
                elif isinstance(predictions[0], dict):
                    pred = predictions[0]
                    scores = pred.get('scores', torch.tensor([])).cpu().numpy()
                    valid_idx = scores > conf_threshold
                    boxes = pred.get('boxes', torch.tensor([])).cpu().numpy()[valid_idx]
                    confs = scores[valid_idx]
                    classes = pred.get('labels', torch.tensor([])).cpu().numpy()[valid_idx]
                else:
                    # 创建模拟检测结果
                    num_boxes = np.random.randint(3, 8)
                    boxes = []
                    confs = []
                    classes = []
                    for _ in range(num_boxes):
                        x1 = np.random.randint(0, width//2)
                        y1 = np.random.randint(0, height//2)
                        x2 = np.random.randint(x1+20, width)
                        y2 = np.random.randint(y1+20, height)
                        boxes.append([x1, y1, x2, y2])
                        confs.append(np.random.uniform(0.5, 0.95))
                        classes.append(np.random.randint(0, len(self.coco_classes)))
                    boxes = np.array(boxes)
                    confs = np.array(confs)
                    classes = np.array(classes)
            else:
                # 创建模拟检测结果
                num_boxes = np.random.randint(3, 8)
                boxes = []
                confs = []
                classes = []
                for _ in range(num_boxes):
                    x1 = np.random.randint(0, width//2)
                    y1 = np.random.randint(0, height//2)
                    x2 = np.random.randint(x1+20, width)
                    y2 = np.random.randint(y1+20, height)
                    boxes.append([x1, y1, x2, y2])
                    confs.append(np.random.uniform(0.5, 0.95))
                    classes.append(np.random.randint(0, len(self.coco_classes)))
                boxes = np.array(boxes)
                confs = np.array(confs)
                classes = np.array(classes)
            
            # 绘制边界框和标签
            if PIL_AVAILABLE and len(boxes) > 0:
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                
                # 尝试加载字体
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    if conf < conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box
                    color = self.detection_colors[int(cls) % len(self.detection_colors)]
                    
                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 绘制标签
                    class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
                    label = f"{class_name}: {conf:.2f}"
                    
                    # 计算文本大小
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 绘制文本背景
                    draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
                    
                    # 绘制文本
                    draw.text((x1+2, y1-text_height-2), label, fill=(255, 255, 255), font=font)
                
                rendered_image = np.array(pil_image)
            elif CV2_AVAILABLE and len(boxes) > 0:
                # 使用OpenCV绘制
                rendered_image = image.copy()
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    if conf < conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = self.detection_colors[int(cls) % len(self.detection_colors)]
                    
                    # 绘制边界框
                    cv2.rectangle(rendered_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    class_name = self.coco_classes[int(cls) % len(self.coco_classes)]
                    label = f"{class_name}: {conf:.2f}"
                    
                    # 获取文本大小
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # 绘制文本背景
                    cv2.rectangle(rendered_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
                    
                    # 绘制文本
                    cv2.putText(rendered_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                rendered_image = image
            
            return rendered_image
            
        except Exception as e:
            self.logger.warning(f"检测结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((640, 640, 3), dtype=np.uint8)
    
    def render_segmentation_result(self, image, predictions, alpha=0.6):
        """渲染分割结果（绘制分割mask）"""
        try:
            # 处理输入图像
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.shape[0] == 3:  # CHW -> HWC
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
            
            # 确保图像在0-255范围内
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            height, width = image.shape[:2]
            
            # 处理预测结果
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() == 4:  # NCHW
                    predictions = predictions.squeeze(0)
                if predictions.dim() == 3 and predictions.shape[0] > 1:  # CHW with multiple classes
                    # 转换为分割mask
                    pred_mask = torch.argmax(torch.softmax(predictions, dim=0), dim=0)
                else:
                    pred_mask = predictions.squeeze()
                
                pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
            else:
                # 创建模拟分割结果
                pred_mask = np.random.randint(0, len(self.cityscapes_classes), (height, width), dtype=np.uint8)
            
            # 调整mask大小以匹配图像
            if pred_mask.shape != (height, width):
                if CV2_AVAILABLE:
                    pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    # 简单的resize
                    pred_mask = np.array(Image.fromarray(pred_mask).resize((width, height), Image.NEAREST))
            
            # 创建彩色分割图
            color_map = np.array([
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
            ], dtype=np.uint8)
            
            # 将分割mask转换为彩色图像
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(len(self.cityscapes_classes)):
                mask_i = (pred_mask == i)
                colored_mask[mask_i] = color_map[i % len(color_map)]
            
            # 混合原图像和分割结果
            rendered_image = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0) if CV2_AVAILABLE else image
            
            # 添加图例（可选）
            if PIL_AVAILABLE:
                pil_image = Image.fromarray(rendered_image)
                draw = ImageDraw.Draw(pil_image)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # 绘制图例
                unique_classes = np.unique(pred_mask)
                legend_y = 10
                for cls_id in unique_classes[:5]:  # 只显示前5个类别
                    if cls_id < len(self.cityscapes_classes):
                        color = tuple(color_map[cls_id % len(color_map)].tolist())
                        class_name = self.cityscapes_classes[cls_id]
                        
                        # 绘制颜色块
                        draw.rectangle([10, legend_y, 30, legend_y+15], fill=color)
                        
                        # 绘制文本
                        draw.text((35, legend_y), class_name, fill=(255, 255, 255), font=font)
                        legend_y += 20
                
                rendered_image = np.array(pil_image)
            
            return rendered_image
            
        except Exception as e:
            self.logger.warning(f"分割结果渲染失败: {e}")
            return image if isinstance(image, np.ndarray) else np.zeros((512, 1024, 3), dtype=np.uint8)


class InteractiveBenchmark:
    def __init__(self):
        self.device = None
        self.model_type = None
        self.model = None
        self.dataset_name = None
        self.dataloader = None
        self.results = []
        self.detailed_results = []  # 存储详细的每帧结果
        
        # 设置日志系统
        self.logger, self.log_filename = setup_logging()
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基准测试工具初始化开始")
        
        # 初始化渲染引擎
        self.rendering_engine = RenderingEngine(self.logger)
        
        # Monitoring
        self.monitoring = True
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_memory = deque(maxlen=1000)
        self.gpu_utilization = deque(maxlen=1000)
        
        self.total_samples = 0
        self.start_time = None
        
        # 导航状态管理
        self.setup_state = {
            'device': False,
            'model_type': False,
            'model': False,
            'dataset': False,
            'samples': False
        }
        
        # 测试样本数量设置
        self.test_samples = 100  # 默认值
        
        # Available models - 扩展模型列表
        self.detection_models = {
            '1': {'name': 'YOLOv8n', 'model': 'yolov8n.pt', 'type': 'yolo'},
            '2': {'name': 'YOLOv8s', 'model': 'yolov8s.pt', 'type': 'yolo'},
            '3': {'name': 'YOLOv8m', 'model': 'yolov8m.pt', 'type': 'yolo'},
            '4': {'name': 'Faster R-CNN ResNet50', 'model': 'fasterrcnn_resnet50_fpn', 'type': 'torchvision'},
            '5': {'name': 'Faster R-CNN MobileNet', 'model': 'fasterrcnn_mobilenet_v3_large_fpn', 'type': 'torchvision'},
            '6': {'name': 'FCOS ResNet50', 'model': 'fcos_resnet50_fpn', 'type': 'torchvision'},
        }
        
        self.classification_models = {
            '1': {'name': 'ResNet18', 'model': 'resnet18', 'type': 'timm'},
            '2': {'name': 'ResNet50', 'model': 'resnet50', 'type': 'timm'},
            '3': {'name': 'EfficientNet-B0', 'model': 'efficientnet_b0', 'type': 'timm'},
            '4': {'name': 'EfficientNet-B3', 'model': 'efficientnet_b3', 'type': 'timm'},
            '5': {'name': 'Vision Transformer', 'model': 'vit_base_patch16_224', 'type': 'timm'},
            '6': {'name': 'MobileNet-V3', 'model': 'mobilenetv3_large_100', 'type': 'timm'},
        }
        
        # 分割模型
        self.segmentation_models = {
            '1': {'name': 'DeepLabV3+ ResNet50', 'model': 'DeepLabV3Plus', 'encoder': 'resnet50', 'type': 'smp'},
            '2': {'name': 'DeepLabV3+ EfficientNet-B0', 'model': 'DeepLabV3Plus', 'encoder': 'efficientnet-b0', 'type': 'smp'},
            '3': {'name': 'UNet ResNet34', 'model': 'Unet', 'encoder': 'resnet34', 'type': 'smp'},
            '4': {'name': 'UNet++ ResNet50', 'model': 'UnetPlusPlus', 'encoder': 'resnet50', 'type': 'smp'},
            '5': {'name': 'PSPNet ResNet50', 'model': 'PSPNet', 'encoder': 'resnet50', 'type': 'smp'},
            '6': {'name': 'FPN ResNet50', 'model': 'FPN', 'encoder': 'resnet50', 'type': 'smp'},
        }
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基准测试工具初始化完成")

    def interactive_setup(self):
        """交互式设置 - 支持返回上一级"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始交互式设置流程")
        
        print("="*60)
        print("深度学习模型基准测试工具")
        print("="*60)
        print("提示：在任何选择阶段输入 'b' 或 'back' 可返回上一步")
        print("="*60)
        
        # 设置流程状态机
        current_step = 'device'
        
        while current_step != 'confirm':
            if current_step == 'device':
                result = self.select_device()
                if result == 'back':
                    print("已在第一步，无法返回")
                    continue
                elif result == 'success':
                    self.setup_state['device'] = True
                    current_step = 'model_type'
                    
            elif current_step == 'model_type':
                result = self.select_model_type()
                if result == 'back':
                    current_step = 'device'
                    self.setup_state['model_type'] = False
                    continue
                elif result == 'success':
                    self.setup_state['model_type'] = True
                    current_step = 'model'
                    
            elif current_step == 'model':
                result = self.select_model()
                if result == 'back':
                    current_step = 'model_type'
                    self.setup_state['model'] = False
                    continue
                elif result == 'success':
                    self.setup_state['model'] = True
                    current_step = 'dataset'
                    
            elif current_step == 'dataset':
                result = self.select_dataset()
                if result == 'back':
                    current_step = 'model'
                    self.setup_state['dataset'] = False
                    continue
                elif result == 'success':
                    self.setup_state['dataset'] = True
                    current_step = 'samples'
                    
            elif current_step == 'samples':
                result = self.select_sample_count()
                if result == 'back':
                    current_step = 'dataset'
                    self.setup_state['samples'] = False
                    continue
                elif result == 'success':
                    self.setup_state['samples'] = True
                    current_step = 'confirm'
        
        # 配置总结和确认
        print("\n" + "="*60)
        print("配置总结：")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.model_type}")
        print(f"模型: {self.model_info['name'] if hasattr(self, 'model_info') else 'Unknown'}")
        print(f"数据集: {self.dataset_name}")
        print(f"测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        print("="*60)
        
        # 记录配置到日志
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 配置完成 - 设备: {self.device}, 模型类型: {self.model_type}, 模型: {self.model_info['name'] if hasattr(self, 'model_info') else 'Unknown'}, 数据集: {self.dataset_name}, 样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        while True:
            confirm = input("\n确认开始测试? (y/n/b-返回设置): ").lower().strip()
            if confirm == 'y':
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户确认开始测试")
                break
            elif confirm == 'n':
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户取消测试")
                print("测试已取消")
                sys.exit(0)
            elif confirm in ['b', 'back']:
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户返回重新设置")
                # 重新开始设置流程
                self.interactive_setup()
                return

    def select_device(self):
        """选择计算设备"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始设备选择")
        
        print("\n1. 选择计算设备:")
        print("1) CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"2) CUDA:{i} - {torch.cuda.get_device_name(i)}")
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到GPU设备: CUDA:{i} - {torch.cuda.get_device_name(i)}")
        else:
            print("   (CUDA不可用)")
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CUDA不可用")
        
        while True:
            choice = input("请选择设备 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.device = 'cpu'
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择设备: {self.device}")
                print(f"已选择设备: {self.device}")
                return 'success'
            elif choice == '2' and torch.cuda.is_available():
                self.device = 'cuda:0'
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择设备: {self.device}")
                print(f"已选择设备: {self.device}")
                return 'success'
            else:
                print("无效选择，请重新输入")

    def select_model_type(self):
        """选择模型类型"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始模型类型选择")
        
        print("\n2. 选择模型类型:")
        print("1) 图像分类 (Classification)")
        if ULTRALYTICS_AVAILABLE or TORCHVISION_DETECTION_AVAILABLE:
            print("2) 目标检测 (Object Detection)")
        else:
            print("2) 目标检测 (需要安装相关依赖)")
        
        if SMP_AVAILABLE:
            print("3) 语义分割 (Semantic Segmentation)")
        else:
            print("3) 语义分割 (需要安装 segmentation-models-pytorch)")
        
        while True:
            choice = input("请选择模型类型 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.model_type = 'classification'
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2' and (ULTRALYTICS_AVAILABLE or TORCHVISION_DETECTION_AVAILABLE):
                self.model_type = 'detection'
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2':
                print("目标检测需要安装相关依赖: pip install ultralytics 或确保 torchvision 版本支持检测模型")
            elif choice == '3' and SMP_AVAILABLE:
                self.model_type = 'segmentation'
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型类型: {self.model_type}")
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '3':
                print("语义分割需要安装: pip install segmentation-models-pytorch")
            else:
                print("无效选择，请重新输入")

    def select_model(self):
        """选择具体模型"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始具体模型选择")
        
        print(f"\n3. 选择{self.model_type}模型:")
        
        if self.model_type == 'classification':
            for key, value in self.classification_models.items():
                print(f"{key}) {value['name']}")
            
            while True:
                choice = input("请选择模型 (输入数字, 'b'返回): ").strip().lower()
                if choice in ['b', 'back']:
                    return 'back'
                elif choice in self.classification_models:
                    selected = self.classification_models[choice]
                    self.model_info = selected
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型: {selected['name']}")
                    print(f"已选择模型: {selected['name']}")
                    return 'success'
                else:
                    print("无效选择，请重新输入")
                    
        elif self.model_type == 'detection':
            available_models = {}
            for key, value in self.detection_models.items():
                if value['type'] == 'yolo' and ULTRALYTICS_AVAILABLE:
                    available_models[key] = value
                    print(f"{key}) {value['name']}")
                elif value['type'] == 'torchvision' and TORCHVISION_DETECTION_AVAILABLE:
                    available_models[key] = value
                    print(f"{key}) {value['name']}")
            
            if not available_models:
                print("没有可用的检测模型，请安装相关依赖")
                return 'back'
            
            while True:
                choice = input("请选择模型 (输入数字, 'b'返回): ").strip().lower()
                if choice in ['b', 'back']:
                    return 'back'
                elif choice in available_models:
                    selected = available_models[choice]
                    self.model_info = selected
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型: {selected['name']}")
                    print(f"已选择模型: {selected['name']}")
                    return 'success'
                else:
                    print("无效选择，请重新输入")
        
        elif self.model_type == 'segmentation':
            for key, value in self.segmentation_models.items():
                print(f"{key}) {value['name']}")
            
            while True:
                choice = input("请选择模型 (输入数字, 'b'返回): ").strip().lower()
                if choice in ['b', 'back']:
                    return 'back'
                elif choice in self.segmentation_models:
                    selected = self.segmentation_models[choice]
                    self.model_info = selected
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择模型: {selected['name']}")
                    print(f"已选择模型: {selected['name']}")
                    return 'success'
                else:
                    print("无效选择，请重新输入")

    def select_dataset(self):
        """选择数据集"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始数据集选择")
        
        print("\n4. 选择数据集:")
        
        if self.model_type == 'classification':
            print("1) MNIST (手写数字, 28x28 -> 224x224)")
            print("2) CIFAR-10 (小物体分类, 32x32 -> 224x224)")
            print("3) ImageNet 验证集样本 (224x224)")
            print("注意：实际数据集大小取决于下一步的样本数量设置")
            
            datasets = {
                '1': {'name': 'MNIST', 'func': self.load_mnist},
                '2': {'name': 'CIFAR-10', 'func': self.load_cifar10},
                '3': {'name': 'ImageNet-Sample', 'func': self.load_imagenet_sample}
            }
            
        elif self.model_type == 'detection':
            print("1) COCO 验证集样本 (需要下载)")
            print("2) KITTI 数据集 (自动驾驶场景)")
            print("3) 预设测试图像")
            print("注意：实际图像数量取决于下一步的样本数量设置")
            
            datasets = {
                '1': {'name': 'COCO-Sample', 'func': self.load_coco_sample},
                '2': {'name': 'KITTI', 'func': self.load_kitti},
                '3': {'name': 'Test-Images', 'func': self.load_test_images}
            }
        
        elif self.model_type == 'segmentation':
            print("1) Cityscapes 数据集 (城市街景分割)")
            print("2) 合成分割数据 (用于测试)")
            print("注意：实际图像数量取决于下一步的样本数量设置")
            
            datasets = {
                '1': {'name': 'Cityscapes', 'func': self.load_cityscapes},
                '2': {'name': 'Synthetic-Segmentation', 'func': self.load_synthetic_segmentation}
            }
        
        while True:
            choice = input("请选择数据集 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice in datasets:
                selected = datasets[choice]
                self.dataset_name = selected['name']
                self.load_dataset_func = selected['func']
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择数据集: {selected['name']}")
                print(f"已选择数据集: {selected['name']}")
                return 'success'
            else:
                print("无效选择，请重新输入")

    def select_sample_count(self):
        """选择测试样本数量"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始样本数量选择")
        
        print("\n5. 选择测试样本数量:")
        print("1) 快速测试 (100 样本)")
        print("2) 中等测试 (500 样本)")
        print("3) 标准测试 (1000 样本)")
        print("4) 大规模测试 (5000 样本)")
        print("5) 全部样本 (使用完整数据集)")
        print("6) 自定义数量")
        
        while True:
            choice = input("请选择测试规模 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.test_samples = 100
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择测试样本数: {self.test_samples}")
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '2':
                self.test_samples = 500
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择测试样本数: {self.test_samples}")
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '3':
                self.test_samples = 1000
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择测试样本数: {self.test_samples}")
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '4':
                self.test_samples = 5000
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择测试样本数: {self.test_samples}")
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '5':
                self.test_samples = -1  # -1 表示全部样本
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择测试样本数: 全部")
                print("已选择测试样本数: 全部")
                return 'success'
            elif choice == '6':
                while True:
                    try:
                        custom_count = input("请输入自定义样本数量 (输入 'b' 返回): ").strip()
                        if custom_count.lower() in ['b', 'back']:
                            break
                        custom_count = int(custom_count)
                        if custom_count > 0:
                            self.test_samples = custom_count
                            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 用户选择自定义测试样本数: {self.test_samples}")
                            print(f"已选择测试样本数: {self.test_samples}")
                            return 'success'
                        else:
                            print("样本数量必须大于0")
                    except ValueError:
                        print("请输入有效的数字")
            else:
                print("无效选择，请重新输入")

    def load_model(self):
        """加载选定的模型"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载模型: {self.model_info['name']}")
        print(f"\n正在加载模型: {self.model_info['name']}...")
        
        try:
            if self.model_type == 'classification':
                if self.model_info['type'] == 'timm':
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用timm加载分类模型: {self.model_info['model']}")
                    self.model = timm.create_model(
                        self.model_info['model'], 
                        pretrained=True,
                        num_classes=1000
                    )
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
            elif self.model_type == 'detection':
                if self.model_info['type'] == 'yolo':
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用YOLO加载检测模型: {self.model_info['model']}")
                    self.model = YOLO(self.model_info['model'])
                elif self.model_info['type'] == 'torchvision':
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用torchvision加载检测模型: {self.model_info['model']}")
                    # 加载torchvision检测模型 - 修复deprecated警告
                    if self.model_info['model'] == 'fasterrcnn_resnet50_fpn':
                        self.model = detection_models.fasterrcnn_resnet50_fpn(weights='DEFAULT')
                    elif self.model_info['model'] == 'fasterrcnn_mobilenet_v3_large_fpn':
                        self.model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
                    elif self.model_info['model'] == 'fcos_resnet50_fpn':
                        self.model = detection_models.fcos_resnet50_fpn(weights='DEFAULT')
                    
                    self.model.eval()
                    self.model = self.model.to(self.device)
            
            elif self.model_type == 'segmentation':
                if self.model_info['type'] == 'smp':
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用segmentation_models_pytorch加载分割模型: {self.model_info['model']}")
                    # 使用segmentation_models_pytorch创建模型
                    model_class = getattr(smp, self.model_info['model'])
                    self.model = model_class(
                        encoder_name=self.model_info['encoder'],
                        encoder_weights='imagenet',
                        classes=19,  # Cityscapes有19个类别
                        activation=None
                    )
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 模型加载成功: {self.model_info['name']}")
            print("模型加载成功!")
            
        except Exception as e:
            self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 模型加载失败: {e}")
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def load_mnist(self):
        """加载MNIST数据集 - 修复通道数问题"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载MNIST数据集")
        print("正在加载MNIST数据集...")
        print("注意：将灰度图像(1通道)转换为RGB图像(3通道)并调整大小到224x224")
        
        # 对于MNIST，需要特殊的预处理：1通道->3通道，28x28->224x224
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为tensor (0-1范围)
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 1通道->3通道
            transforms.Resize((224, 224)),  # 调整大小到224x224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MNIST数据集加载完成，共{len(dataset)}个样本")
        print(f"MNIST数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")
        print("数据预处理：灰度->RGB, 28x28->224x224")

    def load_cifar10(self):
        """加载CIFAR-10数据集 - 修复尺寸问题"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载CIFAR-10数据集")
        print("正在加载CIFAR-10数据集...")
        print("注意：将图像从32x32调整到224x224")
        
        # 对于CIFAR-10，需要调整尺寸：32x32->224x224
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为tensor
            transforms.Resize((224, 224)),  # 调整大小到224x224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CIFAR-10数据集加载完成，共{len(dataset)}个样本")
        print(f"CIFAR-10数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")
        print("数据预处理：32x32->224x224")

    def load_imagenet_sample(self):
        """加载ImageNet样本数据集"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 设置ImageNet样本数据集")
        print("ImageNet样本数据集设置...")
        print("请下载ImageNet验证集到 './data/imagenet/val' 目录")
        print("下载地址: https://www.image-net.org/download.php")
        print("或使用少量样本图像进行测试")
        
        # 创建简单的测试数据
        self.create_synthetic_dataset(224, 1000)

    def load_coco_sample(self):
        """加载COCO样本数据集"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 设置COCO样本数据集")
        print("COCO样本数据集设置...")
        print("请下载COCO 2017验证集到 './data/coco' 目录")
        print("下载地址: http://cocodataset.org/#download")
        
        # 创建简单的测试数据
        self.create_synthetic_detection_data()

    def load_kitti(self):
        """加载KITTI数据集"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载KITTI数据集")
        print("正在加载KITTI数据集...")
        print("KITTI数据集用于自动驾驶场景的目标检测")
        print("请确保KITTI数据集已下载到 './data/kitti' 目录")
        print("下载地址: http://www.cvlibs.net/datasets/kitti/")
        
        # KITTI图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 1248)),  # KITTI典型尺寸调整
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = KITTIDataset(
            root_dir='./data/kitti',
            split='training',
            transform=transform
        )
        
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 设置测试图像列表 - 修复缺失的属性
        if hasattr(dataset, 'image_files'):
            self.test_images = dataset.image_files
        else:
            # 如果没有真实图像文件，创建合成图像列表
            num_images = len(dataset)
            self.test_images = [f"synthetic_kitti_{i:06d}.png" for i in range(num_images)]
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] KITTI数据集加载完成，共{len(dataset)}个样本")
        print(f"KITTI数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")

    def load_test_images(self):
        """加载预设测试图像"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用预设测试图像")
        print("使用预设测试图像...")
        self.create_synthetic_detection_data()

    def load_cityscapes(self):
        """加载Cityscapes分割数据集"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载Cityscapes数据集")
        print("正在加载Cityscapes数据集...")
        print("Cityscapes数据集用于城市街景的语义分割")
        print("请确保Cityscapes数据集已下载到 './data/cityscapes' 目录")
        print("下载地址: https://www.cityscapes-dataset.com/")
        
        # Cityscapes图像预处理 - 修复transform问题
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将numpy数组转换为tensor
            transforms.Resize((512, 1024)),  # 调整到合适大小以加快推理
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 目标变换，处理mask
        def target_transform(mask):
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            elif isinstance(mask, Image.Image):
                mask = torch.from_numpy(np.array(mask)).long()
            if mask.dim() > 2:
                mask = mask.squeeze()
            # 调整大小
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), 
                size=(512, 1024), 
                mode='nearest'
            ).squeeze().long()
            return mask
        
        dataset = CityscapesDataset(
            root_dir='./data/cityscapes',
            split='val',
            transform=transform,
            target_transform=target_transform
        )
        
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cityscapes数据集加载完成，共{len(dataset)}个样本")
        print(f"Cityscapes数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")

    def load_synthetic_segmentation(self):
        """加载合成分割数据"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成分割数据集")
        print("创建合成分割数据集...")
        self.create_synthetic_segmentation_dataset()

    def create_synthetic_dataset(self, img_size, num_classes):
        """创建合成数据集用于测试"""
        # 根据用户选择确定数据集大小
        if self.test_samples == -1:
            dataset_size = 10000  # 全部样本时使用10000作为默认大小
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)")
            print(f"创建合成数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)...")
        else:
            dataset_size = max(self.test_samples, 100)  # 至少100个样本
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)")
            print(f"创建合成数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)...")
        
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, size, img_size=224, num_classes=1000):
                self.size = size
                self.img_size = img_size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机图像 (3通道)
                img = torch.randn(3, self.img_size, self.img_size)
                # 添加ImageNet标准化
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img = normalize(img)
                label = torch.randint(0, self.num_classes, (1,)).item()
                return img, label
        
        dataset = SyntheticDataset(dataset_size, img_size, num_classes)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 合成数据集创建完成")
        print("合成数据集创建完成")

    def create_synthetic_detection_data(self):
        """创建合成检测数据"""
        # 根据用户选择确定测试图像数量
        if self.test_samples == -1:
            num_images = 1000  # 全部样本时使用1000作为默认大小
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成检测数据... ({num_images}张测试图像)")
            print(f"创建合成检测数据... ({num_images}张测试图像)")
        else:
            num_images = max(self.test_samples, 10)  # 至少10张图像
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成检测数据... ({num_images}张测试图像)")
            print(f"创建合成检测数据... ({num_images}张测试图像)")
        
        # 生成测试图像路径列表
        self.test_images = []
        for i in range(num_images):
            # 创建合成图像文件名
            img_path = f"synthetic_test_img_{i:06d}.jpg"
            self.test_images.append(img_path)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建{len(self.test_images)}个测试图像")
        print(f"创建{len(self.test_images)}个测试图像")
        
        # 为了兼容dataloader，也创建一个简单的数据集
        class SyntheticDetectionDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机图像 - 使用numpy数组
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # 转换为tensor
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                
                # 添加标准化
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img = normalize(img)
                
                return img, 0  # 返回图像和虚拟标签
        
        # 创建dataloader（主要用于torchvision模型）
        dataset = SyntheticDetectionDataset(num_images)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def create_synthetic_segmentation_dataset(self):
        """创建合成分割数据集"""
        # 根据用户选择确定数据集大小
        if self.test_samples == -1:
            dataset_size = 500
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)")
            print(f"创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)...")
        else:
            dataset_size = max(self.test_samples, 50)
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)")
            print(f"创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)...")
        
        class SyntheticSegmentationDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # 生成随机图像和分割mask - 使用numpy数组
                img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
                
                # 转换为tensor并添加标准化
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img = normalize(img)
                
                # 生成随机分割mask (19个类别对应Cityscapes)
                mask = torch.randint(0, 19, (512, 1024)).long()
                return img, mask
        
        dataset = SyntheticSegmentationDataset(dataset_size)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 合成分割数据集创建完成")
        print("合成分割数据集创建完成")

    def monitor_resources(self):
        """监控系统资源"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始系统资源监控")
        
        if torch.cuda.is_available() and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                handle = None
        else:
            handle = None
        
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU和内存
                self.cpu_usage.append(process.cpu_percent(interval=0.1))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU监控
                if torch.cuda.is_available():
                    # GPU内存
                    gpu_mem = torch.cuda.memory_allocated(0)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory
                    self.gpu_memory.append((gpu_mem / gpu_total) * 100)
                    
                    # GPU利用率
                    if handle:
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            self.gpu_utilization.append(util.gpu)
                        except:
                            self.gpu_utilization.append(0)
                
                time.sleep(0.1)
                
            except Exception as e:
                break
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 系统资源监控结束")

    def run_classification_benchmark(self):
        """运行分类模型基准测试"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始分类模型基准测试")
        print("\n开始分类模型基准测试...")
        print(f"正在使用模型: {self.model_info['name']}")
        print(f"计划测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        print(f"输入数据格式验证中...")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataloader):
                batch_start = time.time()
                
                # 验证输入数据形状
                if batch_idx == 0:
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 输入数据形状: {data.shape}, 数据类型: {data.dtype}")
                    print(f"输入数据形状: {data.shape}")
                    print(f"数据类型: {data.dtype}")
                    print(f"数据范围: [{data.min().item():.3f}, {data.max().item():.3f}]")
                
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # 推理时间
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 推理过程中出错: {e}")
                    print(f"推理过程中出错: {e}")
                    print(f"输入形状: {data.shape}")
                    print(f"设备: {data.device}")
                    raise e
                
                # 后处理时间（分类任务简单，几乎为0）
                post_start = time.time()
                # 分类任务的后处理很少，这里只是为了保持格式一致
                post_time = (time.time() - post_start) * 1000  # ms
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_classification_result(data, output)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 渲染失败: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 确保时间值合理，避免异常数据
                prep_time = max(prep_time, 0.001)  # 最小0.001ms
                inf_time = max(inf_time, 0.001)
                post_time = max(post_time, 0.001)
                render_time = max(render_time, 0.001)
                batch_time = max(batch_time, 0.001)
                
                # 记录详细结果（每个样本）
                batch_size = len(data)
                if batch_size > 0:  # 防止除零错误
                    for i in range(batch_size):
                        sample_prep_time = prep_time / batch_size
                        sample_inf_time = inf_time / batch_size  
                        sample_post_time = post_time / batch_size
                        sample_render_time = render_time / batch_size
                        sample_total_time = batch_time / batch_size
                        
                        # 确保每个样本时间都是合理的
                        sample_prep_time = max(sample_prep_time, 0.001)
                        sample_inf_time = max(sample_inf_time, 0.001)
                        sample_post_time = max(sample_post_time, 0.001)
                        sample_render_time = max(sample_render_time, 0.001)
                        sample_total_time = max(sample_total_time, 0.001)
                        
                        self.detailed_results.append({
                            'sample_id': self.total_samples + i,
                            'preprocessing_time': sample_prep_time,
                            'inference_time': sample_inf_time,
                            'postprocessing_time': sample_post_time,
                            'rendering_time': sample_render_time,
                            'total_time': sample_total_time
                        })
                
                # 记录汇总时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示和日志记录
                if batch_idx % 10 == 0:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    if self.test_samples == -1:
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 处理进度: {self.total_samples} 样本, 当前FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples} samples... 当前FPS: {fps:.1f}")
                    else:
                        progress = (self.total_samples / self.test_samples) * 100
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 处理进度: {self.total_samples}/{self.test_samples} 样本 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples}/{self.test_samples} samples ({progress:.1f}%)... 当前FPS: {fps:.1f}")
                
                # 根据用户设置限制测试样本数
                if self.test_samples != -1 and self.total_samples >= self.test_samples:
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 达到目标样本数 {self.test_samples}，测试完成")
                    print(f"达到目标样本数 {self.test_samples}，测试完成")
                    break
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分类模型基准测试完成，总计处理 {self.total_samples} 个样本")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }

    def run_detection_benchmark(self):
        """运行检测模型基准测试"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始检测模型基准测试")
        print("\n开始检测模型基准测试...")
        print(f"计划测试图像数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        # 确定实际要测试的图像数量
        if hasattr(self, 'test_images'):
            if self.test_samples == -1:
                num_test_images = len(self.test_images)
            else:
                num_test_images = min(self.test_samples, len(self.test_images))
        else:
            # 如果没有test_images属性，从dataloader获取数量
            dataset_size = len(self.dataloader.dataset)
            if self.test_samples == -1:
                num_test_images = dataset_size
            else:
                num_test_images = min(self.test_samples, dataset_size)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 实际测试图像数: {num_test_images}")
        print(f"实际测试图像数: {num_test_images}")
        
        if self.model_info['type'] == 'yolo':
            # YOLO模型测试
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用YOLO模型进行检测测试")
            for i in range(num_test_images):
                # 创建随机图像进行测试
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # 记录总时间
                total_start = time.time()
                results = self.model(img, device=self.device, verbose=False)
                total_elapsed = (time.time() - total_start) * 1000  # ms
                
                # 获取时间信息
                prep_time = 0.0
                inf_time = total_elapsed  # 默认值
                post_time = 0.0
                
                if hasattr(results[0], 'speed'):
                    speed = results[0].speed
                    prep_time = speed.get('preprocess', 0)
                    inf_time = speed.get('inference', 0)
                    post_time = speed.get('postprocess', 0)
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_detection_result(img, results)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测渲染失败: {e}")
                    render_time = 0.0
                
                # 确保时间值合理
                prep_time = max(prep_time, 0.001)
                inf_time = max(inf_time, 0.001) 
                post_time = max(post_time, 0.001)
                render_time = max(render_time, 0.001)
                total_time = prep_time + inf_time + post_time + render_time
                
                # 如果总时间异常，使用实际测量时间
                if total_time < total_elapsed * 0.5:  # 如果总时间明显小于实际时间
                    total_time = max(total_elapsed + render_time, 0.001)
                    inf_time = total_time - prep_time - post_time - render_time
                    inf_time = max(inf_time, 0.001)
                
                # 记录详细结果
                self.detailed_results.append({
                    'sample_id': i,
                    'preprocessing_time': prep_time,
                    'inference_time': inf_time,
                    'postprocessing_time': post_time,
                    'rendering_time': render_time,
                    'total_time': total_time
                })
                
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                
                self.total_samples += 1
                
                # 进度显示和日志记录
                if i % 10 == 0 or i == num_test_images - 1:
                    fps = 1000.0 / total_time if total_time > 0 else 0
                    progress = ((i + 1) / num_test_images) * 100
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] YOLO检测进度: {i + 1}/{num_test_images} 图像 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                    print(f"Processed {i + 1}/{num_test_images} images ({progress:.1f}%)... 当前FPS: {fps:.1f}")
        
        elif self.model_info['type'] == 'torchvision':
            # Torchvision检测模型测试
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用Torchvision模型进行检测测试")
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.dataloader):
                    batch_start = time.time()
                    
                    # 预处理时间
                    prep_start = time.time()
                    data = data.to(self.device)
                    # 对于检测模型，需要将数据转换为列表格式
                    if data.dim() == 4 and data.size(0) == 1:
                        data_list = [data.squeeze(0)]
                    else:
                        data_list = [img for img in data]
                    prep_time = (time.time() - prep_start) * 1000
                    
                    # 推理时间
                    inf_start = time.time()
                    try:
                        predictions = self.model(data_list)
                        inf_time = (time.time() - inf_start) * 1000
                    except Exception as e:
                        self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Torchvision检测推理过程中出错: {e}")
                        print(f"推理过程中出错: {e}")
                        raise e
                    
                    # 后处理时间（简单计算）
                    post_start = time.time()
                    # Torchvision检测模型的后处理通常包括NMS等，这里简单模拟
                    post_time = (time.time() - post_start) * 1000 + 1.0  # 假设后处理时间
                    
                    # 渲染时间
                    render_start = time.time()
                    try:
                        rendered_image = self.rendering_engine.render_detection_result(data_list[0], predictions)
                        render_time = (time.time() - render_start) * 1000  # ms
                    except Exception as e:
                        self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Torchvision检测渲染失败: {e}")
                        render_time = 0.0
                    
                    total_time = prep_time + inf_time + post_time + render_time
                    
                    # 确保时间值合理
                    prep_time = max(prep_time, 0.001)
                    inf_time = max(inf_time, 0.001)
                    post_time = max(post_time, 0.001)
                    render_time = max(render_time, 0.001)
                    total_time = max(total_time, 0.001)
                    
                    # 记录详细结果
                    self.detailed_results.append({
                        'sample_id': batch_idx,
                        'preprocessing_time': prep_time,
                        'inference_time': inf_time,
                        'postprocessing_time': post_time,
                        'rendering_time': render_time,
                        'total_time': total_time
                    })
                    
                    preprocessing_times.append(prep_time)
                    inference_times.append(inf_time)
                    postprocessing_times.append(post_time)
                    rendering_times.append(render_time)
                    
                    self.total_samples += 1
                    
                    # 进度显示和日志记录
                    if batch_idx % 10 == 0:
                        fps = 1000.0 / total_time if total_time > 0 else 0
                        progress = (self.total_samples / num_test_images) * 100
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Torchvision检测进度: {self.total_samples}/{num_test_images} 图像 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples}/{num_test_images} images ({progress:.1f}%)... 当前FPS: {fps:.1f}")
                    
                    # 限制测试样本数
                    if self.test_samples != -1 and self.total_samples >= self.test_samples:
                        break
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测模型基准测试完成，总计处理 {self.total_samples} 张图像")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times
        }

    def run_segmentation_benchmark(self):
        """运行分割模型基准测试"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始分割模型基准测试")
        print("\n开始分割模型基准测试...")
        print(f"正在使用模型: {self.model_info['name']}")
        print(f"计划测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        rendering_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataloader):
                batch_start = time.time()
                
                # 验证输入数据形状
                if batch_idx == 0:
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割模型输入数据形状: {data.shape}, 目标形状: {target.shape}")
                    print(f"输入数据形状: {data.shape}")
                    print(f"目标形状: {target.shape}")
                    print(f"数据类型: {data.dtype}")
                
                # 预处理时间
                prep_start = time.time()
                data = data.to(self.device)
                prep_time = (time.time() - prep_start) * 1000  # ms
                
                # 推理时间
                inf_start = time.time()
                try:
                    output = self.model(data)
                    inf_time = (time.time() - inf_start) * 1000  # ms
                except Exception as e:
                    self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割推理过程中出错: {e}")
                    print(f"推理过程中出错: {e}")
                    print(f"输入形状: {data.shape}")
                    raise e
                
                # 后处理时间（例如softmax和argmax）
                post_start = time.time()
                if output.dim() > 3:  # 如果输出是logits
                    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                else:
                    pred = output
                post_time = (time.time() - post_start) * 1000  # ms
                
                # 渲染时间
                render_start = time.time()
                try:
                    rendered_image = self.rendering_engine.render_segmentation_result(data, pred)
                    render_time = (time.time() - render_start) * 1000  # ms
                except Exception as e:
                    self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割渲染失败: {e}")
                    render_time = 0.0
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 确保时间值合理
                prep_time = max(prep_time, 0.001)
                inf_time = max(inf_time, 0.001)
                post_time = max(post_time, 0.001)
                render_time = max(render_time, 0.001)
                batch_time = max(batch_time, 0.001)
                
                # 记录详细结果
                batch_size = len(data)
                if batch_size > 0:
                    for i in range(batch_size):
                        sample_prep_time = prep_time / batch_size
                        sample_inf_time = inf_time / batch_size
                        sample_post_time = post_time / batch_size
                        sample_render_time = render_time / batch_size
                        sample_total_time = batch_time / batch_size
                        
                        self.detailed_results.append({
                            'sample_id': self.total_samples + i,
                            'preprocessing_time': sample_prep_time,
                            'inference_time': sample_inf_time,
                            'postprocessing_time': sample_post_time,
                            'rendering_time': sample_render_time,
                            'total_time': sample_total_time
                        })
                
                # 记录汇总时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                postprocessing_times.append(post_time)
                rendering_times.append(render_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示和日志记录
                if batch_idx % 10 == 0:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    if self.test_samples == -1:
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割处理进度: {self.total_samples} 样本, 当前FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples} samples... 当前FPS: {fps:.1f}")
                    else:
                        progress = (self.total_samples / self.test_samples) * 100
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割处理进度: {self.total_samples}/{self.test_samples} 样本 ({progress:.1f}%), 当前FPS: {fps:.1f}")
                        print(f"Processed {self.total_samples}/{self.test_samples} samples ({progress:.1f}%)... 当前FPS: {fps:.1f}")
                
                # 根据用户设置限制测试样本数
                if self.test_samples != -1 and self.total_samples >= self.test_samples:
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 达到目标样本数 {self.test_samples}，分割测试完成")
                    print(f"达到目标样本数 {self.test_samples}，测试完成")
                    break
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分割模型基准测试完成，总计处理 {self.total_samples} 个样本")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times,
            'rendering_times': rendering_times,
            'batch_times': batch_times
        }

    def run_benchmark(self):
        """运行基准测试"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始运行基准测试")
        print("\n开始基准测试...")
        
        # 启动资源监控
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.start_time = time.time()
        
        try:
            if self.model_type == 'classification':
                timing_results = self.run_classification_benchmark()
            elif self.model_type == 'detection':
                timing_results = self.run_detection_benchmark()
            elif self.model_type == 'segmentation':
                timing_results = self.run_segmentation_benchmark()
            
        except KeyboardInterrupt:
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试被用户中断")
            print("\n测试被用户中断")
            return None
        except Exception as e:
            self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试过程中出错: {e}")
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.monitoring = False
            time.sleep(0.5)  # 等待监控线程结束
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基准测试完成，总耗时: {total_time:.2f}秒")
        
        # 计算统计信息
        stats = self.calculate_statistics(timing_results, total_time)
        
        return stats

    def calculate_statistics(self, timing_results, total_time):
        """计算统计信息"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始计算统计信息")
        
        stats = {
            'system_info': {
                'hostname': socket.gethostname(),
                'device': self.device,
                'model_type': self.model_type,
                'model_name': self.model_info['name'],
                'dataset': self.dataset_name,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'performance': {
                'total_samples': self.total_samples,
                'total_time': total_time,
                'throughput': self.total_samples / total_time if total_time > 0 else 0,
                'avg_time_per_sample': (total_time / self.total_samples * 1000) if self.total_samples > 0 else 0
            },
            'timing': {},
            'resources': {
                'cpu': {
                    'min': np.min(self.cpu_usage) if self.cpu_usage else 0,
                    'max': np.max(self.cpu_usage) if self.cpu_usage else 0,
                    'avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                    'std': np.std(self.cpu_usage) if self.cpu_usage else 0
                },
                'memory': {
                    'min': np.min(self.memory_usage) if self.memory_usage else 0,
                    'max': np.max(self.memory_usage) if self.memory_usage else 0,
                    'avg': np.mean(self.memory_usage) if self.memory_usage else 0,
                    'std': np.std(self.memory_usage) if self.memory_usage else 0
                }
            }
        }
        
        # 添加时间统计
        for key, times in timing_results.items():
            if times:
                stats['timing'][key] = {
                    'min': np.min(times),
                    'max': np.max(times),
                    'avg': np.mean(times),
                    'std': np.std(times)
                }
        
        # GPU统计
        if torch.cuda.is_available() and self.gpu_memory:
            stats['resources']['gpu'] = {
                'memory': {
                    'min': np.min(self.gpu_memory),
                    'max': np.max(self.gpu_memory),
                    'avg': np.mean(self.gpu_memory),
                    'std': np.std(self.gpu_memory)
                },
                'utilization': {
                    'min': np.min(self.gpu_utilization) if self.gpu_utilization else 0,
                    'max': np.max(self.gpu_utilization) if self.gpu_utilization else 0,
                    'avg': np.mean(self.gpu_utilization) if self.gpu_utilization else 0,
                    'std': np.std(self.gpu_utilization) if self.gpu_utilization else 0
                }
            }
        
        # 记录关键统计信息到日志
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 统计信息计算完成 - 吞吐量: {stats['performance']['throughput']:.2f} samples/sec")
        
        return stats

    def print_concise_results(self, stats):
        """打印简洁的结果"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始打印测试结果")
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # 基本信息
        print(f"Model: {stats['system_info']['model_name']}")
        print(f"Dataset: {stats['system_info']['dataset']}")
        print(f"Device: {stats['system_info']['device']}")
        print(f"Device Name: {stats['system_info']['device_name']}")
        
        # 性能指标
        print(f"\n{'='*20} PERFORMANCE METRICS {'='*20}")
        print(f"  Samples processed: {stats['performance']['total_samples']}")
        print(f"  Total time: {stats['performance']['total_time']:.2f}s")
        print(f"  Throughput: {stats['performance']['throughput']:.2f} samples/sec")
        print(f"  Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms")
        
        # 时间分解
        if stats['timing']:
            print(f"\n{'='*20} TIMING BREAKDOWN (ms) {'='*20}")
            for stage, data in stats['timing'].items():
                stage_name = stage.replace('_', ' ').title()
                print(f"  {stage_name}:")
                print(f"    Min: {data['min']:.2f}ms")
                print(f"    Max: {data['max']:.2f}ms")
                print(f"    Avg: {data['avg']:.2f}ms ± {data['std']:.2f}")
                print()
        
        # 资源使用
        print(f"{'='*20} RESOURCE UTILIZATION {'='*20}")
        print(f"  CPU Usage:")
        print(f"    Min: {stats['resources']['cpu']['min']:.1f}%")
        print(f"    Max: {stats['resources']['cpu']['max']:.1f}%")
        print(f"    Avg: {stats['resources']['cpu']['avg']:.1f}% ± {stats['resources']['cpu']['std']:.1f}")
        print()
        
        print(f"  Memory Usage:")
        print(f"    Min: {stats['resources']['memory']['min']:.1f}%")
        print(f"    Max: {stats['resources']['memory']['max']:.1f}%")
        print(f"    Avg: {stats['resources']['memory']['avg']:.1f}% ± {stats['resources']['memory']['std']:.1f}")
        print()
        
        if 'gpu' in stats['resources']:
            print(f"  GPU Memory:")
            print(f"    Min: {stats['resources']['gpu']['memory']['min']:.1f}%")
            print(f"    Max: {stats['resources']['gpu']['memory']['max']:.1f}%")
            print(f"    Avg: {stats['resources']['gpu']['memory']['avg']:.1f}% ± {stats['resources']['gpu']['memory']['std']:.1f}")
            print()
            
            print(f"  GPU Utilization:")
            print(f"    Min: {stats['resources']['gpu']['utilization']['min']:.1f}%")
            print(f"    Max: {stats['resources']['gpu']['utilization']['max']:.1f}%")
            print(f"    Avg: {stats['resources']['gpu']['utilization']['avg']:.1f}% ± {stats['resources']['gpu']['utilization']['std']:.1f}")
            print()
        
        # 性能评级
        fps = stats['performance']['throughput']
        if self.model_type == 'classification':
            if fps > 100: rating = "Excellent 🟢"
            elif fps > 50: rating = "Good 🟡"
            elif fps > 10: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        elif self.model_type == 'detection':
            if fps > 30: rating = "Excellent 🟢"
            elif fps > 15: rating = "Good 🟡"
            elif fps > 5: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        else:  # segmentation
            if fps > 20: rating = "Excellent 🟢"
            elif fps > 10: rating = "Good 🟡"
            elif fps > 3: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        
        print(f"{'='*20} OVERALL RATING {'='*25}")
        print(f"  Performance Rating: {rating}")
        print("="*70)
        
        # 记录性能评级到日志
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 性能评级: {rating}, FPS: {fps:.2f}")

    def save_detailed_csv_results(self, stats):
        """保存详细的CSV结果 - 按帧/图像输出"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始保存详细CSV结果")
        
        # 详细结果文件
        detailed_filename = f"{hostname}_{self.model_type}_detailed_{timestamp}.csv"
        
        with open(detailed_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            if self.model_type == 'detection':
                writer.writerow(['Image_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms'])
            elif self.model_type == 'segmentation':
                writer.writerow(['Sample_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms'])
            else:
                writer.writerow(['Sample_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Rendering_Time_ms', 'Total_Time_ms'])
            
            # 写入详细数据
            for result in self.detailed_results:
                writer.writerow([
                    result['sample_id'],
                    f"{result['preprocessing_time']:.4f}",
                    f"{result['inference_time']:.4f}",
                    f"{result['postprocessing_time']:.4f}",
                    f"{result['rendering_time']:.4f}",
                    f"{result['total_time']:.4f}"
                ])
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 详细结果已保存至: {detailed_filename}")
        print(f"\n详细结果已保存至: {detailed_filename}")
        
        # 汇总统计文件
        summary_filename = f"{hostname}_{self.model_type}_summary_{timestamp}.csv"
        
        with open(summary_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 系统信息部分
            writer.writerow(['=== SYSTEM INFORMATION ==='])
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Hostname', stats['system_info']['hostname'], ''])
            writer.writerow(['Model Type', stats['system_info']['model_type'], ''])
            writer.writerow(['Model Name', stats['system_info']['model_name'], ''])
            writer.writerow(['Dataset', stats['system_info']['dataset'], ''])
            writer.writerow(['Device', stats['system_info']['device'], ''])
            writer.writerow(['Device Name', stats['system_info']['device_name'], ''])
            writer.writerow(['PyTorch Version', stats['system_info']['torch_version'], ''])
            writer.writerow(['CUDA Available', stats['system_info']['cuda_available'], ''])
            writer.writerow(['Test Start Time', time.strftime('%Y-%m-%d %H:%M:%S'), ''])
            writer.writerow([])  # 空行分隔
            
            # 性能指标部分
            writer.writerow(['=== PERFORMANCE METRICS ==='])
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Total Samples', stats['performance']['total_samples'], 'samples'])
            writer.writerow(['Total Time', f"{stats['performance']['total_time']:.4f}", 'seconds'])
            writer.writerow(['Throughput', f"{stats['performance']['throughput']:.4f}", 'samples/sec'])
            writer.writerow(['Avg Time per Sample', f"{stats['performance']['avg_time_per_sample']:.4f}", 'ms'])
            writer.writerow([])  # 空行分隔
            
            # 时间分解部分
            if stats['timing']:
                writer.writerow(['=== TIMING BREAKDOWN ==='])
                writer.writerow(['Stage', 'Min (ms)', 'Max (ms)', 'Avg (ms)', 'Std (ms)'])
                for stage, data in stats['timing'].items():
                    stage_name = stage.replace('_', ' ').title()
                    writer.writerow([
                        stage_name,
                        f"{data['min']:.4f}",
                        f"{data['max']:.4f}",
                        f"{data['avg']:.4f}",
                        f"{data['std']:.4f}"
                    ])
                writer.writerow([])  # 空行分隔
            
            # 资源使用部分
            writer.writerow(['=== RESOURCE UTILIZATION ==='])
            writer.writerow(['Resource', 'Min (%)', 'Max (%)', 'Avg (%)'])
            writer.writerow(['CPU Usage', 
                           f"{stats['resources']['cpu']['min']:.2f}", 
                           f"{stats['resources']['cpu']['max']:.2f}", 
                           f"{stats['resources']['cpu']['avg']:.2f}"])
            writer.writerow(['Memory Usage', 
                           f"{stats['resources']['memory']['min']:.2f}", 
                           f"{stats['resources']['memory']['max']:.2f}", 
                           f"{stats['resources']['memory']['avg']:.2f}"])
            
            if 'gpu' in stats['resources']:
                writer.writerow(['GPU Memory', 
                               f"{stats['resources']['gpu']['memory']['min']:.2f}", 
                               f"{stats['resources']['gpu']['memory']['max']:.2f}", 
                               f"{stats['resources']['gpu']['memory']['avg']:.2f}"])
                writer.writerow(['GPU Utilization', 
                               f"{stats['resources']['gpu']['utilization']['min']:.2f}", 
                               f"{stats['resources']['gpu']['utilization']['max']:.2f}", 
                               f"{stats['resources']['gpu']['utilization']['avg']:.2f}"])
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 汇总结果已保存至: {summary_filename}")
        print(f"汇总结果已保存至: {summary_filename}")
        return detailed_filename, summary_filename

    def create_detailed_timing_plot(self, timestamp, hostname):
        """创建详细的每帧速度分析折线图"""
        if not self.detailed_results or len(self.detailed_results) < 10:
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据不足，跳过详细时间折线图生成")
            print("数据不足，跳过详细时间折线图生成")
            return None
            
        try:
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始生成详细速度分析折线图")
            print("正在生成详细速度分析折线图...")
            
            # 提取数据并计算速度指标
            sample_ids = [r['sample_id'] for r in self.detailed_results]
            total_times = [r['total_time'] for r in self.detailed_results]
            inf_times = [r['inference_time'] for r in self.detailed_results]
            prep_times = [r['preprocessing_time'] for r in self.detailed_results]
            post_times = [r['postprocessing_time'] for r in self.detailed_results]
            render_times = [r['rendering_time'] for r in self.detailed_results]
            
            # 计算FPS（每秒帧数）- 避免除零错误
            fps_total = []
            fps_inference = []
            throughput = []
            
            for i, (total_time, inf_time) in enumerate(zip(total_times, inf_times)):
                # 确保时间值合理，避免无穷大
                total_time = max(total_time, 0.001)  # 最小0.001ms
                inf_time = max(inf_time, 0.001)
                
                # FPS = 1000 / time_ms （从毫秒转换）
                fps_t = min(1000.0 / total_time, 10000)  # 限制最大FPS为10000
                fps_i = min(1000.0 / inf_time, 10000)
                
                fps_total.append(fps_t)
                fps_inference.append(fps_i)
                
                # 吞吐量（样本/秒）
                throughput.append(fps_t)
            
            # 计算平均值
            avg_fps_total = np.mean(fps_total)
            avg_fps_inference = np.mean(fps_inference)
            avg_throughput = np.mean(throughput)
            
            # 创建图表
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    plt.style.use('default')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f'Per-Frame Speed Analysis: {self.model_info["name"]} on {self.dataset_name}', fontsize=16)
            
            # 上图：FPS性能图
            ax1.plot(sample_ids, fps_total, label='Total FPS', color='blue', alpha=0.7, linewidth=1.5)
            ax1.plot(sample_ids, fps_inference, label='Inference FPS', color='red', alpha=0.7, linewidth=1.5)
            
            # 添加平均值线
            ax1.axhline(y=avg_fps_total, color='blue', linestyle='--', alpha=0.8, linewidth=2, 
                       label=f'Avg Total FPS: {avg_fps_total:.1f}')
            ax1.axhline(y=avg_fps_inference, color='red', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Avg Inference FPS: {avg_fps_inference:.1f}')
            
            ax1.set_xlabel('Sample/Image ID')
            ax1.set_ylabel('FPS (Frames Per Second)')
            ax1.set_title('Processing Speed per Frame')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 添加统计信息文本
            stats_text = f"""Speed Statistics:
Total FPS: {avg_fps_total:.1f} ± {np.std(fps_total):.1f}
Inference FPS: {avg_fps_inference:.1f} ± {np.std(fps_inference):.1f}
Min FPS: {np.min(fps_total):.1f}
Max FPS: {np.max(fps_total):.1f}"""
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 下图：时间分解堆叠图
            ax2.fill_between(sample_ids, 0, prep_times, label='Preprocessing', alpha=0.8, color='lightblue')
            ax2.fill_between(sample_ids, prep_times, [p + i for p, i in zip(prep_times, inf_times)], 
                            label='Inference', alpha=0.8, color='lightcoral')
            ax2.fill_between(sample_ids, [p + i for p, i in zip(prep_times, inf_times)], 
                            [p + i + post for p, i, post in zip(prep_times, inf_times, post_times)], 
                            label='Postprocessing', alpha=0.8, color='lightgreen')
            ax2.fill_between(sample_ids, [p + i + post for p, i, post in zip(prep_times, inf_times, post_times)], 
                            [p + i + post + r for p, i, post, r in zip(prep_times, inf_times, post_times, render_times)], 
                            label='Rendering', alpha=0.8, color='gold')
            
            ax2.set_xlabel('Sample/Image ID')
            ax2.set_ylabel('Time (ms)')
            ax2.set_title('Time Breakdown per Sample')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            timing_plot_filename = f"{hostname}_{self.model_type}_speed_analysis_{timestamp}.png"
            plt.savefig(timing_plot_filename, format='png', dpi=300, bbox_inches='tight')
            
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 详细速度分析图表已保存至: {timing_plot_filename}")
            print(f"详细速度分析图表已保存至: {timing_plot_filename}")
            
            plt.close(fig)
            return timing_plot_filename
            
        except Exception as e:
            self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建详细速度折线图时出错: {e}")
            print(f"创建详细速度折线图时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_visualizations(self, stats, csv_filenames):
        """创建可视化图表"""
        try:
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始生成可视化图表")
            print("正在生成可视化图表...")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            hostname = socket.gethostname()
            
            # 首先生成详细时间折线图
            timing_plot_file = self.create_detailed_timing_plot(timestamp, hostname)
            
            # 设置图表样式 - 修复样式名称
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    plt.style.use('default')
                    
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Benchmark Results Summary: {stats["system_info"]["model_name"]} on {stats["system_info"]["dataset"]}', fontsize=16)
            
            # 1. 时间分解饼图
            if stats['timing']:
                timing_data = [(k.replace('_', ' ').title(), v['avg']) for k, v in stats['timing'].items()]
                labels, values = zip(*timing_data)
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Timing Breakdown')
            else:
                ax1.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Timing Breakdown')
            
            # 2. 资源利用率柱状图
            resources = ['CPU', 'Memory']
            usage = [stats['resources']['cpu']['avg'], stats['resources']['memory']['avg']]
            errors = [stats['resources']['cpu']['std'], stats['resources']['memory']['std']]
            
            if 'gpu' in stats['resources']:
                resources.extend(['GPU Memory', 'GPU Util'])
                usage.extend([stats['resources']['gpu']['memory']['avg'], 
                             stats['resources']['gpu']['utilization']['avg']])
                errors.extend([stats['resources']['gpu']['memory']['std'],
                              stats['resources']['gpu']['utilization']['std']])
            
            bars = ax2.bar(resources, usage, yerr=errors, capsize=5,
                          color=['skyblue', 'lightgreen', 'orange', 'red'][:len(resources)])
            ax2.set_title('Resource Utilization (with std dev)')
            ax2.set_ylabel('Usage (%)')
            ax2.set_ylim(0, 100)
            
            # 添加数值标签
            for bar, val, err in zip(bars, usage, errors):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 1, 
                        f'{val:.1f}%', ha='center', va='bottom')
            
            # 3. 性能时间分布 - 如果有详细结果的话
            if self.detailed_results and len(self.detailed_results) > 10:
                # 绘制时间分布直方图
                total_times = [r['total_time'] for r in self.detailed_results]
                ax3.hist(total_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                ax3.set_xlabel('Total Time per Sample (ms)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Processing Time Distribution')
                ax3.axvline(np.mean(total_times), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(total_times):.2f}ms')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Insufficient data\nfor distribution plot', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Processing Time Distribution')
            
            # 4. 系统信息和性能总结
            if 'gpu' in stats['resources']:
                system_text = f"""System Information:
Device: {stats['system_info']['device']}
Model: {stats['system_info']['model_name']}
Dataset: {stats['system_info']['dataset']}

Performance Summary:
Throughput: {stats['performance']['throughput']:.2f} samples/sec
Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms
Total samples: {stats['performance']['total_samples']}

Resource Usage (Avg ± Std):
CPU: {stats['resources']['cpu']['avg']:.1f}% ± {stats['resources']['cpu']['std']:.1f}%
Memory: {stats['resources']['memory']['avg']:.1f}% ± {stats['resources']['memory']['std']:.1f}%
GPU Mem: {stats['resources']['gpu']['memory']['avg']:.1f}% ± {stats['resources']['gpu']['memory']['std']:.1f}%
GPU Util: {stats['resources']['gpu']['utilization']['avg']:.1f}% ± {stats['resources']['gpu']['utilization']['std']:.1f}%"""
            else:
                system_text = f"""System Information:
Device: {stats['system_info']['device']}
Model: {stats['system_info']['model_name']}
Dataset: {stats['system_info']['dataset']}

Performance Summary:
Throughput: {stats['performance']['throughput']:.2f} samples/sec
Avg time/sample: {stats['performance']['avg_time_per_sample']:.2f}ms
Total samples: {stats['performance']['total_samples']}

Resource Usage (Avg ± Std):
CPU: {stats['resources']['cpu']['avg']:.1f}% ± {stats['resources']['cpu']['std']:.1f}%
Memory: {stats['resources']['memory']['avg']:.1f}% ± {stats['resources']['memory']['std']:.1f}%"""
            
            ax4.text(0.05, 0.95, system_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('System Info & Performance Summary')
            
            plt.tight_layout()
            
            # 保存总结图表
            summary_plot_filename = f"{hostname}_{self.model_type}_summary_{timestamp}.png"
            plt.savefig(summary_plot_filename, format='png', dpi=300, bbox_inches='tight')
            
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 性能总结图表已保存至: {summary_plot_filename}")
            print(f"性能总结图表已保存至: {summary_plot_filename}")
            
            # 关闭图形以释放内存
            plt.close(fig)
            
            # 返回生成的图表文件名
            generated_plots = [summary_plot_filename]
            if timing_plot_file:
                generated_plots.append(timing_plot_file)
            
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 可视化图表生成完成，共生成 {len(generated_plots)} 个图表文件")
            
            return generated_plots
            
        except Exception as e:
            self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建可视化时出错: {e}")
            print(f"创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            return []


def main():
    """主函数"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 深度学习基准测试工具启动")
    
    # 检查依赖
    missing_deps = []
    
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics (pip install ultralytics)")
    
    if not NVML_AVAILABLE:
        missing_deps.append("nvidia-ml-py3 (pip install nvidia-ml-py3)")
    
    if not SMP_AVAILABLE:
        missing_deps.append("segmentation-models-pytorch (pip install segmentation-models-pytorch)")
    
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow (pip install Pillow)")
    
    try:
        import timm
    except ImportError:
        missing_deps.append("timm (pip install timm)")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        missing_deps.append("matplotlib seaborn (pip install matplotlib seaborn)")
    
    if missing_deps:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到缺失依赖")
        print("建议安装以下依赖以获得完整功能:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()
    
    # 创建基准测试工具
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化基准测试工具")
    benchmark = InteractiveBenchmark()
    
    # 交互式设置
    benchmark.interactive_setup()
    
    # 加载数据集
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载数据集")
    benchmark.load_dataset_func()
    
    # 加载模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始加载模型")
    benchmark.load_model()
    
    # 运行基准测试
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始运行基准测试")
    stats = benchmark.run_benchmark()
    
    if stats:
        # 打印简洁结果
        benchmark.print_concise_results(stats)
        
        # 保存详细CSV结果
        csv_filenames = benchmark.save_detailed_csv_results(stats)
        
        # 创建可视化
        plot_files = benchmark.create_visualizations(stats, csv_filenames)
        
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试完成!")
        print(f"日志文件: {benchmark.log_filename}")
        print(f"详细结果文件: {csv_filenames[0]}")
        print(f"汇总结果文件: {csv_filenames[1]}")
        if plot_files:
            print("生成的图表文件:")
            for plot_file in plot_files:
                print(f"  - {plot_file}")
        
        # 记录最终完成状态到日志
        benchmark.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 所有测试和输出生成完成")
        benchmark.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 生成文件: 日志-{benchmark.log_filename}, CSV-{csv_filenames[0]}, 汇总-{csv_filenames[1]}, 图表-{len(plot_files)}个")
        
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
