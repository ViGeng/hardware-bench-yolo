#!/usr/bin/env python3
"""
Dataset module - contains loading and preprocessing functions for various datasets
"""

import os
import logging
import time
import random
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

class InfiniteRandomDataset(Dataset):
    """Infinite random sampling dataset wrapper"""
    def __init__(self, original_dataset, target_length):
        self.original_dataset = original_dataset
        self.target_length = target_length
        self.original_length = len(original_dataset)
        
    def __len__(self):
        return self.target_length
    
    def __getitem__(self, idx):
        # Randomly select a sample from the original dataset
        random_idx = random.randint(0, self.original_length - 1)
        return self.original_dataset[random_idx]

class KITTIDataset(Dataset):
    """KITTI dataset class"""
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
        # Image path
        self.image_dir = self.root_dir / split / 'image_2'
        
        # Get all image files
        if self.image_dir.exists():
            self.image_files = sorted(list(self.image_dir.glob('*.png')))
            self.logger.info(f"KITTI dataset found {len(self.image_files)} image files")
        else:
            # If no KITTI data exists, create synthetic data
            self.logger.warning(f"KITTI data path does not exist: {self.image_dir}")
            self.logger.info("Will use synthetic data for testing")
            self.image_files = [f"synthetic_kitti_{i:06d}.png" for i in range(1000)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # Create synthetic KITTI-style image (375x1242 is typical KITTI size)
            img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        else:
            # Load real image
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
            else:
                img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, 0

class CityscapesDataset(Dataset):
    """Cityscapes dataset class (for segmentation)"""
    def __init__(self, root_dir, split='val', transform=None, target_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
        # Image and label paths
        self.image_dir = self.root_dir / 'leftImg8bit' / split
        self.label_dir = self.root_dir / 'gtFine' / split
        
        # Get image files
        if self.image_dir.exists():
            self.image_files = []
            for city_dir in self.image_dir.iterdir():
                if city_dir.is_dir():
                    self.image_files.extend(list(city_dir.glob('*_leftImg8bit.png')))
            self.image_files = sorted(self.image_files)
            self.logger.info(f"Cityscapes dataset found {len(self.image_files)} image files")
        else:
            self.logger.warning(f"Cityscapes data path does not exist: {self.image_dir}")
            self.logger.info("Will use synthetic data for testing")
            self.image_files = [f"synthetic_cityscapes_{i:06d}.png" for i in range(500)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if isinstance(self.image_files[idx], str) and 'synthetic' in self.image_files[idx]:
            # Create synthetic Cityscapes-style image (512x1024, reduced size for faster processing)
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)  # 19 classes
        else:
            img_path = self.image_files[idx]
            if PIL_AVAILABLE:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
                # Try to find corresponding label file
                label_path = str(img_path).replace('leftImg8bit', 'gtFine_labelIds').replace('leftImg8bit.png', 'gtFine_labelIds.png')
                if os.path.exists(label_path):
                    mask = Image.open(label_path)
                    mask = np.array(mask)
                else:
                    mask = np.random.randint(0, 19, (img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
                mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return img, mask

class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for classification testing"""
    def __init__(self, size, img_size=224, num_classes=1000):
        self.size = size
        self.img_size = img_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image (3 channels)
        img = torch.randn(3, self.img_size, self.img_size)
        # Add ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label

class SyntheticDetectionDataset(torch.utils.data.Dataset):
    """Synthetic detection dataset"""
    def __init__(self, size):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Add normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        
        return img, 0

class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """Synthetic segmentation dataset"""
    def __init__(self, size):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image and segmentation mask
        img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        
        # Convert to tensor and add normalization
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        
        # Generate random segmentation mask (19 classes corresponding to Cityscapes)
        mask = torch.randint(0, 19, (512, 1024)).long()
        return img, mask

class DatasetLoader:
    """Dataset loader class"""
    
    def __init__(self, test_samples=100):
        self.test_samples = test_samples
        self.logger = logging.getLogger(__name__)
    
    def _create_infinite_dataloader_if_needed(self, dataset, batch_size=1, shuffle=False):
        """Create infinite dataloader if needed"""
        dataset_size = len(dataset)
        
        # If test sample count is larger than dataset size, use infinite random sampling
        if self.test_samples != -1 and self.test_samples > dataset_size:
            self.logger.info(f"Test sample count ({self.test_samples}) is larger than dataset size ({dataset_size}), using random repeat sampling")
            print(f"Test sample count ({self.test_samples}) is larger than dataset size ({dataset_size})")
            print("Will use random repeat sampling to reach target sample count")
            
            # Use infinite random dataset wrapper
            infinite_dataset = InfiniteRandomDataset(dataset, self.test_samples)
            dataloader = DataLoader(infinite_dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            # Normal case, use standard dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def load_mnist(self):
        """Load MNIST dataset - fix channel count issue"""
        self.logger.info("Starting to load MNIST dataset")
        print("Loading MNIST dataset...")
        print("Note: Converting grayscale images (1 channel) to RGB images (3 channels) and resizing to 224x224")
        
        # For MNIST, special preprocessing is needed: 1 channel->3 channels, 28x28->224x224
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info(f"MNIST dataset loading completed, total {len(dataset)} samples")
        print(f"MNIST dataset loading completed, total {len(dataset)} samples")
        print(f"Will test {self.test_samples if self.test_samples != -1 else len(dataset)} samples according to user settings")
        
        return dataloader
    
    def load_cifar10(self):
        """Load CIFAR-10 dataset - fix size issue"""
        self.logger.info("Starting to load CIFAR-10 dataset")
        print("Loading CIFAR-10 dataset...")
        print("Note: Resizing images from 32x32 to 224x224")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info(f"CIFAR-10 dataset loading completed, total {len(dataset)} samples")
        print(f"CIFAR-10 dataset loading completed, total {len(dataset)} samples")
        
        return dataloader
    
    def load_kitti(self):
        """Load KITTI dataset"""
        self.logger.info("Starting to load KITTI dataset")
        print("Loading KITTI dataset...")
        
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
        
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info(f"KITTI dataset loading completed, total {len(dataset)} samples")
        print(f"KITTI dataset loading completed, total {len(dataset)} samples")
        
        return dataloader
    
    def load_cityscapes(self):
        """Load Cityscapes segmentation dataset"""
        self.logger.info("Starting to load Cityscapes dataset")
        print("Loading Cityscapes dataset...")
        
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
            # Resize
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
        
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info(f"Cityscapes dataset loading completed, total {len(dataset)} samples")
        print(f"Cityscapes dataset loading completed, total {len(dataset)} samples")
        
        return dataloader
    
    def create_synthetic_classification_dataset(self, img_size=224, num_classes=1000):
        """Create synthetic classification dataset"""
        if self.test_samples == -1:
            dataset_size = 10000
        else:
            dataset_size = max(self.test_samples, 100)
        
        self.logger.info(f"Creating synthetic classification dataset ({img_size}x{img_size}, {num_classes} classes, {dataset_size} samples)")
        print(f"Creating synthetic classification dataset ({img_size}x{img_size}, {num_classes} classes, {dataset_size} samples)...")
        
        dataset = SyntheticDataset(dataset_size, img_size, num_classes)
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info("Synthetic classification dataset creation completed")
        print("Synthetic classification dataset creation completed")
        
        return dataloader
    
    def create_synthetic_detection_dataset(self):
        """Create synthetic detection dataset"""
        if self.test_samples == -1:
            num_images = 1000
        else:
            num_images = max(self.test_samples, 10)
        
        self.logger.info(f"Creating synthetic detection dataset ({num_images} test images)")
        print(f"Creating synthetic detection dataset ({num_images} test images)")
        
        dataset = SyntheticDetectionDataset(num_images)
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        # Generate test image path list (for compatibility)
        test_images = [f"synthetic_test_img_{i:06d}.jpg" for i in range(num_images)]
        
        self.logger.info("Synthetic detection dataset creation completed")
        print("Synthetic detection dataset creation completed")
        
        return dataloader, test_images
    
    def create_synthetic_segmentation_dataset(self):
        """Create synthetic segmentation dataset"""
        if self.test_samples == -1:
            dataset_size = 500
        else:
            dataset_size = max(self.test_samples, 50)
        
        self.logger.info(f"Creating synthetic segmentation dataset (512x1024, 19 classes, {dataset_size} samples)")
        print(f"Creating synthetic segmentation dataset (512x1024, 19 classes, {dataset_size} samples)")
        
        dataset = SyntheticSegmentationDataset(dataset_size)
        dataloader = self._create_infinite_dataloader_if_needed(dataset, batch_size=1, shuffle=True)
        
        self.logger.info("Synthetic segmentation dataset creation completed")
        print("Synthetic segmentation dataset creation completed")
        
        return dataloader