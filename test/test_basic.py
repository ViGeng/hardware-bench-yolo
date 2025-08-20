#!/usr/bin/env python3
"""Basic tests for dl_benchmark package."""

import unittest
import torch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dl_benchmark.utils import check_dependencies, get_system_info
from dl_benchmark.config import CLASSIFICATION_MODELS, DETECTION_MODELS, SEGMENTATION_MODELS


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the benchmark tool."""
    
    def test_dependencies_check(self):
        """Test dependency checking functionality."""
        deps = check_dependencies()
        self.assertIsInstance(deps, dict)
        self.assertIn('pil', deps)
        self.assertIn('tqdm', deps)
        
    def test_system_info(self):
        """Test system information gathering."""
        info = get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn('hostname', info)
        self.assertIn('torch_version', info)
        self.assertIn('cuda_available', info)
        
    def test_model_configs(self):
        """Test model configuration dictionaries."""
        self.assertIsInstance(CLASSIFICATION_MODELS, dict)
        self.assertIsInstance(DETECTION_MODELS, dict)
        self.assertIsInstance(SEGMENTATION_MODELS, dict)
        
        # Check if model configs have required keys
        for model in CLASSIFICATION_MODELS.values():
            self.assertIn('name', model)
            self.assertIn('model', model)
            
    def test_torch_installation(self):
        """Test PyTorch installation and basic functionality."""
        # Test tensor creation
        x = torch.randn(2, 3)
        self.assertEqual(x.shape, (2, 3))
        
        # Test CUDA availability (may be False, but should not error)
        cuda_available = torch.cuda.is_available()
        self.assertIsInstance(cuda_available, bool)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_classification_models(self):
        """Test classification model configurations."""
        for key, model in CLASSIFICATION_MODELS.items():
            self.assertIn('name', model)
            self.assertIn('model', model)
            self.assertIn('type', model)
            self.assertEqual(model['type'], 'timm')
            
    def test_detection_models(self):
        """Test detection model configurations."""
        for key, model in DETECTION_MODELS.items():
            self.assertIn('name', model)
            self.assertIn('model', model)
            self.assertIn('type', model)
            self.assertIn(model['type'], ['yolo', 'torchvision'])
            
    def test_segmentation_models(self):
        """Test segmentation model configurations."""
        for key, model in SEGMENTATION_MODELS.items():
            self.assertIn('name', model)
            self.assertIn('model', model)
            self.assertIn('encoder', model)
            self.assertIn('type', model)
            self.assertEqual(model['type'], 'smp')


if __name__ == '__main__':
    unittest.main()