#!/usr/bin/env python3
"""
Enhanced Deep Learning Benchmark Tool - 增强版深度学习基准测试工具
支持多种模型类型和数据集的交互式基准测试

主要功能：
1. 交互式设备选择 (CPU/GPU) - 支持返回上一级
2. 模型类型选择 (Detection/Classification) - 支持返回上一级
3. 数据集选择 (MNIST/CIFAR-10/COCO/ImageNet) - 支持返回上一级
4. 自定义样本数量选择 - 从100到全部数据集
5. 详细性能统计和CSV输出 - 按帧/图像输出详细时间信息
6. 结果可视化

修复内容：
- 修复MNIST数据集的1通道到3通道转换问题
- 添加了适当的图像尺寸调整
- 改进了数据预处理流程
- 添加了完全可自定义的样本数量功能

Author: Enhanced for professional AI benchmarking
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
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Deep Learning Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


class GrayscaleToRGB(object):
    """将灰度图像转换为RGB图像的变换"""
    def __call__(self, img):
        if img.shape[0] == 1:  # 如果是单通道
            return img.repeat(3, 1, 1)  # 复制到3个通道
        return img


class InteractiveBenchmark:
    def __init__(self):
        self.device = None
        self.model_type = None
        self.model = None
        self.dataset_name = None
        self.dataloader = None
        self.results = []
        self.detailed_results = []  # 新增：存储详细的每帧结果
        
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
        
        # Available models
        self.detection_models = {
            '1': {'name': 'YOLOv8n', 'model': 'yolov8n.pt', 'type': 'yolo'},
            '2': {'name': 'YOLOv8s', 'model': 'yolov8s.pt', 'type': 'yolo'},
            '3': {'name': 'YOLOv8m', 'model': 'yolov8m.pt', 'type': 'yolo'},
        }
        
        self.classification_models = {
            '1': {'name': 'ResNet18', 'model': 'resnet18', 'type': 'timm'},
            '2': {'name': 'ResNet50', 'model': 'resnet50', 'type': 'timm'},
            '3': {'name': 'EfficientNet-B0', 'model': 'efficientnet_b0', 'type': 'timm'},
            '4': {'name': 'EfficientNet-B3', 'model': 'efficientnet_b3', 'type': 'timm'},
            '5': {'name': 'Vision Transformer', 'model': 'vit_base_patch16_224', 'type': 'timm'},
            '6': {'name': 'MobileNet-V3', 'model': 'mobilenetv3_large_100', 'type': 'timm'},
        }

    def interactive_setup(self):
        """交互式设置 - 支持返回上一级"""
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
        
        while True:
            confirm = input("\n确认开始测试? (y/n/b-返回设置): ").lower().strip()
            if confirm == 'y':
                break
            elif confirm == 'n':
                print("测试已取消")
                sys.exit(0)
            elif confirm in ['b', 'back']:
                # 重新开始设置流程
                self.interactive_setup()
                return

    def select_device(self):
        """选择计算设备"""
        print("\n1. 选择计算设备:")
        print("1) CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"2) CUDA:{i} - {torch.cuda.get_device_name(i)}")
        else:
            print("   (CUDA不可用)")
        
        while True:
            choice = input("请选择设备 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.device = 'cpu'
                print(f"已选择设备: {self.device}")
                return 'success'
            elif choice == '2' and torch.cuda.is_available():
                self.device = 'cuda:0'
                print(f"已选择设备: {self.device}")
                return 'success'
            else:
                print("无效选择，请重新输入")

    def select_model_type(self):
        """选择模型类型"""
        print("\n2. 选择模型类型:")
        print("1) 图像分类 (Classification)")
        if ULTRALYTICS_AVAILABLE:
            print("2) 目标检测 (Object Detection)")
        else:
            print("2) 目标检测 (需要安装 ultralytics)")
        
        while True:
            choice = input("请选择模型类型 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice == '1':
                self.model_type = 'classification'
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2' and ULTRALYTICS_AVAILABLE:
                self.model_type = 'detection'
                print(f"已选择模型类型: {self.model_type}")
                return 'success'
            elif choice == '2':
                print("目标检测需要安装 ultralytics: pip install ultralytics")
            else:
                print("无效选择，请重新输入")

    def select_model(self):
        """选择具体模型"""
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
                    print(f"已选择模型: {selected['name']}")
                    return 'success'
                else:
                    print("无效选择，请重新输入")
                    
        elif self.model_type == 'detection':
            for key, value in self.detection_models.items():
                print(f"{key}) {value['name']}")
            
            while True:
                choice = input("请选择模型 (输入数字, 'b'返回): ").strip().lower()
                if choice in ['b', 'back']:
                    return 'back'
                elif choice in self.detection_models:
                    selected = self.detection_models[choice]
                    self.model_info = selected
                    print(f"已选择模型: {selected['name']}")
                    return 'success'
                else:
                    print("无效选择，请重新输入")

    def select_dataset(self):
        """选择数据集"""
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
            print("2) 预设测试图像")
            print("注意：实际图像数量取决于下一步的样本数量设置")
            
            datasets = {
                '1': {'name': 'COCO-Sample', 'func': self.load_coco_sample},
                '2': {'name': 'Test-Images', 'func': self.load_test_images}
            }
        
        while True:
            choice = input("请选择数据集 (输入数字, 'b'返回): ").strip().lower()
            if choice in ['b', 'back']:
                return 'back'
            elif choice in datasets:
                selected = datasets[choice]
                self.dataset_name = selected['name']
                self.load_dataset_func = selected['func']
                print(f"已选择数据集: {selected['name']}")
                return 'success'
            else:
                print("无效选择，请重新输入")

    def select_sample_count(self):
        """选择测试样本数量"""
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
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '2':
                self.test_samples = 500
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '3':
                self.test_samples = 1000
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '4':
                self.test_samples = 5000
                print(f"已选择测试样本数: {self.test_samples}")
                return 'success'
            elif choice == '5':
                self.test_samples = -1  # -1 表示全部样本
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
        print(f"\n正在加载模型: {self.model_info['name']}...")
        
        try:
            if self.model_type == 'classification':
                if self.model_info['type'] == 'timm':
                    self.model = timm.create_model(
                        self.model_info['model'], 
                        pretrained=True,
                        num_classes=1000
                    )
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
            elif self.model_type == 'detection':
                if self.model_info['type'] == 'yolo':
                    self.model = YOLO(self.model_info['model'])
                    
            print("模型加载成功!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    def load_mnist(self):
        """加载MNIST数据集 - 修复通道数问题"""
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
        print(f"MNIST数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")
        print("数据预处理：灰度->RGB, 28x28->224x224")

    def load_cifar10(self):
        """加载CIFAR-10数据集 - 修复尺寸问题"""
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
        print(f"CIFAR-10数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")
        print("数据预处理：32x32->224x224")

    def load_imagenet_sample(self):
        """加载ImageNet样本数据集"""
        print("ImageNet样本数据集设置...")
        print("请下载ImageNet验证集到 './data/imagenet/val' 目录")
        print("下载地址: https://www.image-net.org/download.php")
        print("或使用少量样本图像进行测试")
        
        # 创建简单的测试数据
        self.create_synthetic_dataset(224, 1000)

    def load_coco_sample(self):
        """加载COCO样本数据集"""
        print("COCO样本数据集设置...")
        print("请下载COCO 2017验证集到 './data/coco' 目录")
        print("下载地址: http://cocodataset.org/#download")
        
        # 创建简单的测试数据
        self.create_synthetic_detection_data()

    def load_test_images(self):
        """加载预设测试图像"""
        print("使用预设测试图像...")
        self.create_synthetic_detection_data()

    def create_synthetic_dataset(self, img_size, num_classes):
        """创建合成数据集用于测试"""
        # 根据用户选择确定数据集大小
        if self.test_samples == -1:
            dataset_size = 10000  # 全部样本时使用10000作为默认大小
            print(f"创建合成数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)...")
        else:
            dataset_size = max(self.test_samples, 100)  # 至少100个样本
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
        print("合成数据集创建完成")

    def create_synthetic_detection_data(self):
        """创建合成检测数据"""
        # 根据用户选择确定测试图像数量
        if self.test_samples == -1:
            num_images = 1000  # 全部样本时使用1000作为默认大小
            print(f"创建合成检测数据... ({num_images}张测试图像)")
        else:
            num_images = max(self.test_samples, 10)  # 至少10张图像
            print(f"创建合成检测数据... ({num_images}张测试图像)")
        
        # 生成测试图像路径
        self.test_images = []
        for i in range(num_images):
            # 创建随机图像并保存
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = f"./temp_test_img_{i}.jpg"
            
            # 这里应该保存图像，但为了简化我们只保存路径
            self.test_images.append(img_path)
        
        print(f"创建{len(self.test_images)}个测试图像")

    def monitor_resources(self):
        """监控系统资源"""
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

    def run_classification_benchmark(self):
        """运行分类模型基准测试"""
        print("\n开始分类模型基准测试...")
        print(f"正在使用模型: {self.model_info['name']}")
        print(f"计划测试样本数: {self.test_samples if self.test_samples != -1 else '全部'}")
        print(f"输入数据格式验证中...")
        
        batch_times = []
        preprocessing_times = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataloader):
                batch_start = time.time()
                
                # 验证输入数据形状
                if batch_idx == 0:
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
                    print(f"推理过程中出错: {e}")
                    print(f"输入形状: {data.shape}")
                    print(f"设备: {data.device}")
                    raise e
                
                batch_time = (time.time() - batch_start) * 1000  # ms
                
                # 确保时间值合理，避免异常数据
                prep_time = max(prep_time, 0.001)  # 最小0.001ms
                inf_time = max(inf_time, 0.001)
                batch_time = max(batch_time, 0.001)
                
                # 记录详细结果（每个样本）
                batch_size = len(data)
                if batch_size > 0:  # 防止除零错误
                    for i in range(batch_size):
                        sample_prep_time = prep_time / batch_size
                        sample_inf_time = inf_time / batch_size  
                        sample_total_time = batch_time / batch_size
                        
                        # 确保每个样本时间都是合理的
                        sample_prep_time = max(sample_prep_time, 0.001)
                        sample_inf_time = max(sample_inf_time, 0.001)
                        sample_total_time = max(sample_total_time, 0.001)
                        
                        self.detailed_results.append({
                            'sample_id': self.total_samples + i,
                            'preprocessing_time': sample_prep_time,
                            'inference_time': sample_inf_time,
                            'postprocessing_time': 0.0,  # 分类任务无后处理
                            'total_time': sample_total_time
                        })
                
                # 记录汇总时间
                preprocessing_times.append(prep_time)
                inference_times.append(inf_time)
                batch_times.append(batch_time)
                
                self.total_samples += len(data)
                
                # 进度显示
                if batch_idx % 10 == 0:
                    fps = 1000.0 / (batch_time / len(data)) if batch_time > 0 else 0
                    if self.test_samples == -1:
                        print(f"Processed {self.total_samples} samples... 当前FPS: {fps:.1f}")
                    else:
                        progress = (self.total_samples / self.test_samples) * 100
                        print(f"Processed {self.total_samples}/{self.test_samples} samples ({progress:.1f}%)... 当前FPS: {fps:.1f}")
                
                # 根据用户设置限制测试样本数
                if self.test_samples != -1 and self.total_samples >= self.test_samples:
                    print(f"达到目标样本数 {self.test_samples}，测试完成")
                    break
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'batch_times': batch_times
        }

    def run_detection_benchmark(self):
        """运行检测模型基准测试"""
        print("\n开始检测模型基准测试...")
        print(f"计划测试图像数: {self.test_samples if self.test_samples != -1 else len(self.test_images)}")
        
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        
        # 确定实际要测试的图像数量
        if self.test_samples == -1:
            num_test_images = len(self.test_images)
        else:
            num_test_images = min(self.test_samples, len(self.test_images))
        
        print(f"实际测试图像数: {num_test_images}")
        
        # 使用合成图像进行测试
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
            
            # 确保时间值合理
            prep_time = max(prep_time, 0.001)
            inf_time = max(inf_time, 0.001) 
            post_time = max(post_time, 0.001)
            total_time = prep_time + inf_time + post_time
            
            # 如果总时间异常，使用实际测量时间
            if total_time < total_elapsed * 0.5:  # 如果总时间明显小于实际时间
                total_time = max(total_elapsed, 0.001)
                inf_time = total_time - prep_time - post_time
                inf_time = max(inf_time, 0.001)
            
            # 记录详细结果
            self.detailed_results.append({
                'sample_id': i,
                'preprocessing_time': prep_time,
                'inference_time': inf_time,
                'postprocessing_time': post_time,
                'total_time': total_time
            })
            
            preprocessing_times.append(prep_time)
            inference_times.append(inf_time)
            postprocessing_times.append(post_time)
            
            self.total_samples += 1
            
            # 进度显示
            if i % 10 == 0 or i == num_test_images - 1:
                fps = 1000.0 / total_time if total_time > 0 else 0
                progress = ((i + 1) / num_test_images) * 100
                print(f"Processed {i + 1}/{num_test_images} images ({progress:.1f}%)... 当前FPS: {fps:.1f}")
        
        return {
            'preprocessing_times': preprocessing_times,
            'inference_times': inference_times,
            'postprocessing_times': postprocessing_times
        }

    def run_benchmark(self):
        """运行基准测试"""
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
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            return None
        except Exception as e:
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.monitoring = False
            time.sleep(0.5)  # 等待监控线程结束
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # 计算统计信息
        stats = self.calculate_statistics(timing_results, total_time)
        
        return stats

    def calculate_statistics(self, timing_results, total_time):
        """计算统计信息"""
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
        
        return stats

    def print_concise_results(self, stats):
        """打印简洁的结果"""
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
        else:  # detection
            if fps > 30: rating = "Excellent 🟢"
            elif fps > 15: rating = "Good 🟡"
            elif fps > 5: rating = "Fair 🟠"
            else: rating = "Slow 🔴"
        
        print(f"{'='*20} OVERALL RATING {'='*25}")
        print(f"  Performance Rating: {rating}")
        print("="*70)

    def save_detailed_csv_results(self, stats):
        """保存详细的CSV结果 - 按帧/图像输出"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        
        # 详细结果文件
        detailed_filename = f"{hostname}_{self.model_type}_detailed_{timestamp}.csv"
        
        with open(detailed_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            if self.model_type == 'detection':
                writer.writerow(['Image_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Total_Time_ms'])
            else:
                writer.writerow(['Sample_ID', 'Preprocessing_Time_ms', 'Inference_Time_ms', 'Postprocessing_Time_ms', 'Total_Time_ms'])
            
            # 写入详细数据
            for result in self.detailed_results:
                writer.writerow([
                    result['sample_id'],
                    f"{result['preprocessing_time']:.4f}",
                    f"{result['inference_time']:.4f}",
                    f"{result['postprocessing_time']:.4f}",
                    f"{result['total_time']:.4f}"
                ])
        
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
        
        print(f"汇总结果已保存至: {summary_filename}")
        return detailed_filename, summary_filename

    def create_detailed_timing_plot(self, timestamp, hostname):
        """创建详细的每帧速度分析折线图"""
        if not self.detailed_results or len(self.detailed_results) < 10:
            print("数据不足，跳过详细时间折线图生成")
            return None
            
        try:
            print("正在生成详细速度分析折线图...")
            
            # 提取数据并计算速度指标
            sample_ids = [r['sample_id'] for r in self.detailed_results]
            total_times = [r['total_time'] for r in self.detailed_results]
            inf_times = [r['inference_time'] for r in self.detailed_results]
            prep_times = [r['preprocessing_time'] for r in self.detailed_results]
            
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
            
            # 下图：吞吐量和性能稳定性
            ax2.plot(sample_ids, throughput, label='Throughput (samples/sec)', color='green', alpha=0.6, linewidth=1)
            
            # 计算移动平均（窗口大小为10）
            if len(throughput) >= 10:
                window_size = min(10, len(throughput) // 5)
                moving_avg = []
                for i in range(len(throughput)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(throughput), i + window_size // 2 + 1)
                    moving_avg.append(np.mean(throughput[start_idx:end_idx]))
                
                ax2.plot(sample_ids, moving_avg, label=f'Moving Average (window={window_size})', 
                        color='orange', linewidth=2)
            
            # 添加平均吞吐量线
            ax2.axhline(y=avg_throughput, color='green', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Avg Throughput: {avg_throughput:.1f} samples/sec')
            
            # 性能稳定性区间（±1标准差）
            std_throughput = np.std(throughput)
            ax2.fill_between(sample_ids, 
                           avg_throughput - std_throughput, 
                           avg_throughput + std_throughput, 
                           alpha=0.2, color='gray', label=f'±1σ: {std_throughput:.1f}')
            
            ax2.set_xlabel('Sample/Image ID')
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Processing Throughput and Stability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 性能等级标记
            if self.model_type == 'classification':
                performance_levels = [
                    (100, 'Excellent', 'green'),
                    (50, 'Good', 'yellow'), 
                    (10, 'Fair', 'orange'),
                    (0, 'Slow', 'red')
                ]
            else:  # detection
                performance_levels = [
                    (30, 'Excellent', 'green'),
                    (15, 'Good', 'yellow'),
                    (5, 'Fair', 'orange'), 
                    (0, 'Slow', 'red')
                ]
            
            # 添加性能等级线
            for level, label, color in performance_levels[:-1]:  # 跳过最后的0线
                ax2.axhline(y=level, color=color, linestyle=':', alpha=0.5, linewidth=1)
                ax2.text(max(sample_ids) * 0.95, level + std_throughput * 0.1, 
                        label, color=color, fontweight='bold', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图表
            timing_plot_filename = f"{hostname}_{self.model_type}_speed_analysis_{timestamp}.png"
            plt.savefig(timing_plot_filename, format='png', dpi=300, bbox_inches='tight')
            print(f"详细速度分析图表已保存至: {timing_plot_filename}")
            
            plt.close(fig)
            return timing_plot_filename
            
        except Exception as e:
            print(f"创建详细速度折线图时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_visualizations(self, stats, csv_filenames):
        """创建可视化图表"""
        try:
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
            print(f"性能总结图表已保存至: {summary_plot_filename}")
            
            # 关闭图形以释放内存
            plt.close(fig)
            
            # 返回生成的图表文件名
            generated_plots = [summary_plot_filename]
            if timing_plot_file:
                generated_plots.append(timing_plot_file)
            
            return generated_plots
            
        except Exception as e:
            print(f"创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            return []


def main():
    """主函数"""
    # 检查依赖
    missing_deps = []
    
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics (pip install ultralytics)")
    
    if not NVML_AVAILABLE:
        missing_deps.append("nvidia-ml-py3 (pip install nvidia-ml-py3)")
    
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
        print("建议安装以下依赖以获得完整功能:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()
    
    # 创建基准测试工具
    benchmark = InteractiveBenchmark()
    
    # 交互式设置
    benchmark.interactive_setup()
    
    # 加载数据集
    benchmark.load_dataset_func()
    
    # 加载模型
    benchmark.load_model()
    
    # 运行基准测试
    stats = benchmark.run_benchmark()
    
    if stats:
        # 打印简洁结果
        benchmark.print_concise_results(stats)
        
        # 保存详细CSV结果
        csv_filenames = benchmark.save_detailed_csv_results(stats)
        
        # 创建可视化
        plot_files = benchmark.create_visualizations(stats, csv_filenames)
        
        print("\n测试完成!")
        print(f"详细结果文件: {csv_filenames[0]}")
        print(f"汇总结果文件: {csv_filenames[1]}")
        if plot_files:
            print("生成的图表文件:")
            for plot_file in plot_files:
                print(f"  - {plot_file}")
    else:
        print("测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
