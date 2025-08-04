#!/usr/bin/env python3
"""
数据集模块 - 包含各种数据集的加载和预处理功能
"""

import os
import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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
            self.logger.info(f"KITTI数据集找到 {len(self.image_files)} 个图像文件")
        else:
            # 如果没有KITTI数据，创建合成数据
            self.logger.warning(f"KITTI数据路径不存在: {self.image_dir}")
            self.logger.info("将使用合成数据进行测试")
            self.image_files = [f"synthetic_kitti_{i:06d}.png" for i in range(1000)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # 创建合成KITTI样式图像 (375x1242是KITTI的典型尺寸)
            img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        else:
            # 加载真实图像
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
            else:
                img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, 0

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
            self.logger.info(f"Cityscapes数据集找到 {len(self.image_files)} 个图像文件")
        else:
            self.logger.warning(f"Cityscapes数据路径不存在: {self.image_dir}")
            self.logger.info("将使用合成数据进行测试")
            self.image_files = [f"synthetic_cityscapes_{i:06d}.png" for i in range(500)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # 创建合成Cityscapes样式图像 (512x1024，减小尺寸以加快处理)
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)  # 19个类别
        else:
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
                # 尝试找到对应的标签文件
                label_path = str(img_path).replace('leftImg8bit', 'gtFine_labelIds').replace('leftImg8bit.png', 'gtFine_labelIds.png')
                if os.path.exists(label_path):
                    mask = Image.open(label_path)
                    mask = np.array(mask)
                else:
                    mask = np.random.randint(0, 19, (img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
                mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return img, mask

class SyntheticDataset(torch.utils.data.Dataset):
    """合成数据集用于分类测试"""
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

class SyntheticDetectionDataset(torch.utils.data.Dataset):
    """合成检测数据集"""
    def __init__(self, size):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机图像
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 转换为tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 添加标准化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        
        return img, 0

class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """合成分割数据集"""
    def __init__(self, size):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机图像和分割mask
        img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        
        # 转换为tensor并添加标准化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        
        # 生成随机分割mask (19个类别对应Cityscapes)
        mask = torch.randint(0, 19, (512, 1024)).long()
        return img, mask

class DatasetLoader:
    """数据集加载器类"""
    
    def __init__(self, test_samples=100):
        self.test_samples = test_samples
        self.logger = logging.getLogger(__name__)
    
    def load_mnist(self):
        """加载MNIST数据集 - 修复通道数问题"""
        self.logger.info("开始加载MNIST数据集")
        print("正在加载MNIST数据集...")
        print("注意：将灰度图像(1通道)转换为RGB图像(3通道)并调整大小到224x224")
        
        # 对于MNIST，需要特殊的预处理：1通道->3通道，28x28->224x224
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"MNIST数据集加载完成，共{len(dataset)}个样本")
        print(f"MNIST数据集加载完成，共{len(dataset)}个样本")
        print(f"将根据用户设置测试 {self.test_samples if self.test_samples != -1 else len(dataset)} 个样本")
        
        return dataloader
    
    def load_cifar10(self):
        """加载CIFAR-10数据集 - 修复尺寸问题"""
        self.logger.info("开始加载CIFAR-10数据集")
        print("正在加载CIFAR-10数据集...")
        print("注意：将图像从32x32调整到224x224")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"CIFAR-10数据集加载完成，共{len(dataset)}个样本")
        print(f"CIFAR-10数据集加载完成，共{len(dataset)}个样本")
        
        return dataloader
    
    def load_kitti(self):
        """加载KITTI数据集"""
        self.logger.info("开始加载KITTI数据集")
        print("正在加载KITTI数据集...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 1248)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = KITTIDataset(
            root_dir='./data/kitti',
            split='training',
            transform=transform
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"KITTI数据集加载完成，共{len(dataset)}个样本")
        print(f"KITTI数据集加载完成，共{len(dataset)}个样本")
        
        return dataloader
    
    def load_cityscapes(self):
        """加载Cityscapes分割数据集"""
        self.logger.info("开始加载Cityscapes数据集")
        print("正在加载Cityscapes数据集...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 1024)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info(f"Cityscapes数据集加载完成，共{len(dataset)}个样本")
        print(f"Cityscapes数据集加载完成，共{len(dataset)}个样本")
        
        return dataloader
    
    def create_synthetic_classification_dataset(self, img_size=224, num_classes=1000):
        """创建合成分类数据集"""
        if self.test_samples == -1:
            dataset_size = 10000
        else:
            dataset_size = max(self.test_samples, 100)
        
        self.logger.info(f"创建合成分类数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)")
        print(f"创建合成分类数据集 ({img_size}x{img_size}, {num_classes}类, {dataset_size}个样本)...")
        
        dataset = SyntheticDataset(dataset_size, img_size, num_classes)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info("合成分类数据集创建完成")
        print("合成分类数据集创建完成")
        
        return dataloader
    
    def create_synthetic_detection_dataset(self):
        """创建合成检测数据集"""
        if self.test_samples == -1:
            num_images = 1000
        else:
            num_images = max(self.test_samples, 10)
        
        self.logger.info(f"创建合成检测数据集 ({num_images}张测试图像)")
        print(f"创建合成检测数据集 ({num_images}张测试图像)")
        
        dataset = SyntheticDetectionDataset(num_images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 生成测试图像路径列表（用于兼容性）
        test_images = [f"synthetic_test_img_{i:06d}.jpg" for i in range(num_images)]
        
        self.logger.info("合成检测数据集创建完成")
        print("合成检测数据集创建完成")
        
        return dataloader, test_images
    
    def create_synthetic_segmentation_dataset(self):
        """创建合成分割数据集"""
        if self.test_samples == -1:
            dataset_size = 500
        else:
            dataset_size = max(self.test_samples, 50)
        
        self.logger.info(f"创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)")
        print(f"创建合成分割数据集 (512x1024, 19类, {dataset_size}个样本)")
        
        dataset = SyntheticSegmentationDataset(dataset_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.logger.info("合成分割数据集创建完成")
        print("合成分割数据集创建完成")
        
        return dataloader