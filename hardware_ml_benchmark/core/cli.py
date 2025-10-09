#!/usr/bin/env python3
"""
Command line interface module - handles command line arguments and parameter validation
"""

import argparse
import logging
import os
import sys

import torch

from hardware_ml_benchmark.core.config import (CLASSIFICATION_MODELS,
                                               DETECTION_MODELS,
                                               SAMPLE_OPTIONS,
                                               SEGMENTATION_MODELS)
from hardware_ml_benchmark.models.models import validate_model_availability


class CommandLineInterface:
    """Command line interface class"""
    
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.logger = logging.getLogger(__name__)
    
    def create_parser(self):
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description='Deep Learning Model Benchmark Tool',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Example usage:
  # Test ResNet18 classification model using CPU
  python -m hardware_ml_benchmark.main --device cpu --task classification --model resnet18 --dataset MNIST --samples 100
  
  # Test YOLOv8 detection model using GPU  
  python -m hardware_ml_benchmark.main --device cuda:0 --task detection --model yolov8n --dataset Test-Images --samples 500
  
  # Test with local model file
  python -m hardware_ml_benchmark.main --device auto --task detection --model ~/models/my_yolo.pt --dataset Test-Images --samples 100
  
  # Test with model from URL
  python -m hardware_ml_benchmark.main --device auto --task detection --model https://example.com/models/yolo.pt --dataset Test-Images --samples 100
  
  # Test segmentation model with auto device selection
  python -m hardware_ml_benchmark.main --device auto --task segmentation --model unet_resnet34 --dataset Synthetic-Segmentation --samples 200
  
  # Generate visualization plots
  python -m hardware_ml_benchmark.main --device auto --task classification --model resnet18 --dataset MNIST --samples 100 --plot
  
  # List available models
  python -m hardware_ml_benchmark.main --list-models
  
  # List available datasets  
  python -m hardware_ml_benchmark.main --list-datasets
            """
        )
        
        # Add various command line arguments
        parser.add_argument('--device', 
                          choices=['cpu', 'cuda:0', 'auto'], 
                          default='auto',
                          help='Computing device selection:\n'
                               '  cpu - Force CPU computation\n'
                               '  cuda:0 - Force GPU computation\n'
                               '  auto - Auto selection (use GPU if available, otherwise CPU)\n'
                               '  (default: auto)')
        
        parser.add_argument('--task', 
                          choices=['classification', 'detection', 'segmentation'],
                          help='Task type (required)\n'
                               '  classification - Image classification tasks\n'
                               '  detection - Object detection tasks\n'
                               '  segmentation - Semantic segmentation tasks')
        
        parser.add_argument('--model',
                          help='Model name, local path, or URL (required)\n'
                               '  Examples:\n'
                               '    - Model name: resnet18, yolov8n\n'
                               '    - Local path: ~/models/my_model.pt, ./checkpoints/best.pth\n'
                               '    - URL: https://example.com/models/model.pt\n'
                               '  Use --list-models to see available built-in models')
        
        parser.add_argument('--dataset',
                          help='Dataset name (required, use --list-datasets to see available datasets)')
        
        parser.add_argument('--samples', 
                          type=int, 
                          default=100,
                          help='Number of test samples (default: 100, -1 means all)')
        
        parser.add_argument('--batch-size',
                          type=int,
                          default=1,
                          help='Batch size (default: 1)')
        
        parser.add_argument('--output-dir',
                          default='./results',
                          help='Output directory (default: ./results)')
        
        parser.add_argument('--plot',
                          action='store_true',
                          help='Generate visualization plots (disabled by default)')
        
        parser.add_argument('--quiet',
                          action='store_true',
                          help='Quiet mode, reduce output')
        
        parser.add_argument('--list-models',
                          action='store_true',
                          help='List all available models')
        
        parser.add_argument('--list-datasets',
                          action='store_true',
                          help='List all available datasets')
        
        # Monitoring related parameters
        parser.add_argument('--disable-gpu-monitor',
                          action='store_true',
                          help='Disable detailed GPU monitoring')
        
        parser.add_argument('--monitor-interval',
                          type=float,
                          default=0.1,
                          help='Monitoring sampling interval (seconds) (default: 0.1)')
        
        parser.add_argument('--monitor-samples',
                          type=int,
                          default=1000,
                          help='Maximum monitoring samples (default: 1000)')
        
        return parser
    
    def list_available_models(self):
        """List all available models"""
        print("Available Models List:")
        print("="*60)
        print("\nNote: You can also specify:")
        print("  - Local model file: ~/models/my_model.pt, ./checkpoints/best.pth")
        print("  - Model URL: https://example.com/models/model.pt")
        print()
        
        # Classification models
        print("\nImage Classification Models (Classification):")
        print("Usage: --task classification --model <model_name>")
        if validate_model_availability('classification', self.dependencies):
            for key, model in CLASSIFICATION_MODELS.items():
                status = "✓" if self.dependencies['timm'] else "✗"
                print(f"  {status} {model['model']:<25} - {model['name']}")
        else:
            print("  Required installation: pip install timm")
        
        # Detection models
        print("\nObject Detection Models (Detection):")
        print("Usage: --task detection --model <model_name>")
        if validate_model_availability('detection', self.dependencies):
            for key, model in DETECTION_MODELS.items():
                if model['type'] == 'yolo':
                    status = "✓" if self.dependencies['ultralytics'] else "✗"
                    req = "ultralytics" if not self.dependencies['ultralytics'] else ""
                    # Show both formats: with .pt and without .pt
                    model_name = model['model']
                    if model_name.endswith('.pt'):
                        model_id = f"{model_name} or {model_name[:-3]}"
                    else:
                        model_id = model_name
                elif model['type'] == 'torchvision':
                    status = "✓" if self.dependencies['torchvision_detection'] else "✗"
                    req = "torchvision (latest version)" if not self.dependencies['torchvision_detection'] else ""
                    model_id = model['model'].replace('_', '-')
                else:
                    status = "✗"
                    req = "unknown dependency"
                    model_id = model['model']
                
                print(f"  {status} {model_id:<30} - {model['name']}")
                if req:
                    print(f"    Required installation: pip install {req}")
        else:
            print("  Required installation: pip install ultralytics or update torchvision")
        
        # Segmentation models
        print("\nSemantic Segmentation Models (Segmentation):")
        print("Usage: --task segmentation --model <model_name>")
        if validate_model_availability('segmentation', self.dependencies):
            for key, model in SEGMENTATION_MODELS.items():
                status = "✓" if self.dependencies['smp'] else "✗"
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                print(f"  {status} {model_id:<25} - {model['name']}")
        else:
            print("  Required installation: pip install segmentation-models-pytorch")
        
        # Computing device selection description
        print("\nComputing Device Selection (--device):")
        print("="*40)
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ cpu      - Use CPU computation (always available)")
            print(f"  ✓ cuda:0   - Use GPU computation: {device_name}")
            print(f"  ✓ auto     - Auto selection (recommended, will choose: GPU)")
        else:
            print(f"  ✓ cpu      - Use CPU computation (always available)")
            print(f"  ✗ cuda:0   - Use GPU computation (CUDA unavailable)")
            print(f"  ✓ auto     - Auto selection (recommended, will choose: CPU)")
    
    def list_available_datasets(self):
        """List all available datasets"""
        print("Available Datasets List:")
        print("="*60)
        
        print("\nClassification Datasets (--task classification):")
        print("  MNIST              - Handwritten digit recognition (28x28 -> 224x224)")
        print("  CIFAR-10           - Small object classification (32x32 -> 224x224)")
        print("  ImageNet-Sample    - Synthetic ImageNet samples (224x224)")
        
        print("\nDetection Datasets (--task detection):")
        print("  COCO-Sample        - Synthetic COCO samples")
        print("  KITTI              - Autonomous driving scene data")
        print("  Test-Images        - Preset test images")
        
        print("\nSegmentation Datasets (--task segmentation):")
        print("  Cityscapes         - Urban street scene segmentation")
        print("  Synthetic-Segmentation - Synthetic segmentation data")
        
        print("\nUsage Examples:")
        print("  # Built-in model")
        print("  python -m hardware_ml_benchmark.main --task classification --model resnet18 --dataset MNIST --device auto")
        print("  \n  # Local model file")
        print("  python -m hardware_ml_benchmark.main --task detection --model ~/models/my_yolo.pt --dataset Test-Images --device cuda:0")
        print("  \n  # Model from URL")
        print("  python -m hardware_ml_benchmark.main --task detection --model https://example.com/yolo.pt --dataset Test-Images --device auto")
        print("  \n  # With visualization plots")
        print("  python -m hardware_ml_benchmark.main --task classification --model resnet18 --dataset MNIST --device auto --plot")
    
    def _is_url(self, path_str):
        """Check if string is a URL"""
        return path_str.startswith('http://') or path_str.startswith('https://')
    
    def _is_local_path(self, path_str):
        """Check if string is a local path"""
        # Check for path indicators: /, \, ~, or file extensions
        path_indicators = ['/', '\\', '~']
        file_extensions = ['.pt', '.pth', '.ckpt', '.weights', '.bin', '.safetensors']
        
        # If it contains path separators or starts with ~
        if any(indicator in path_str for indicator in path_indicators):
            return True
        
        # If it ends with a model file extension
        if any(path_str.endswith(ext) for ext in file_extensions):
            return True
        
        # If it starts with . (relative path)
        if path_str.startswith('.'):
            return True
            
        return False
    
    def validate_args(self, args):
        """Validate command line arguments"""
        errors = []
        
        # Validate device
        if args.device == 'cuda:0' and not torch.cuda.is_available():
            errors.append("Specified CUDA device but CUDA is not available. Please use --device cpu or --device auto")
        
        # Validate task availability
        if args.task:
            if not validate_model_availability(args.task, self.dependencies):
                if args.task == 'classification':
                    errors.append("Classification models unavailable, required installation: pip install timm")
                elif args.task == 'detection':
                    errors.append("Detection models unavailable, required installation: pip install ultralytics")
                elif args.task == 'segmentation':
                    errors.append("Segmentation models unavailable, required installation: pip install segmentation-models-pytorch")
        
        # Validate model name (relaxed for custom paths/URLs)
        if args.task and args.model:
            # If it's a URL or path, do basic validation
            if self._is_url(args.model):
                # URL validation - just check if it looks like a valid URL
                if not (args.model.startswith('http://') or args.model.startswith('https://')):
                    errors.append(f"Invalid URL format: {args.model}")
            elif self._is_local_path(args.model):
                # Local path validation - expand and check if exists
                expanded_path = os.path.expanduser(args.model)
                abs_path = os.path.abspath(expanded_path)
                if not os.path.exists(abs_path):
                    errors.append(f"Model file not found: {abs_path}")
                    errors.append(f"Original path: {args.model}")
            else:
                # Otherwise, validate as model name
                valid_model = self._validate_model_name(args.task, args.model)
                if not valid_model:
                    errors.append(f"Invalid model name: {args.model}")
                    errors.append(f"Please use --list-models to see available models for {args.task} type")
                    errors.append(f"Or specify a local path (~/models/model.pt) or URL (https://...)")
        
        # Validate dataset name
        if args.task and args.dataset:
            valid_dataset = self._validate_dataset_name(args.task, args.dataset)
            if not valid_dataset:
                errors.append(f"Invalid dataset name: {args.dataset}")
                errors.append(f"Please use --list-datasets to see available datasets for {args.task} type")
        
        # Validate sample count
        if args.samples < -1 or args.samples == 0:
            errors.append("Sample count must be positive or -1 (meaning all)")
        
        return errors
    
    def _validate_model_name(self, task, model_name):
        """Validate if model name is valid"""
        if task == 'classification':
            valid_models = [model['model'] for model in CLASSIFICATION_MODELS.values()]
            return model_name in valid_models
        
        elif task == 'detection':
            valid_models = []
            for model in DETECTION_MODELS.values():
                if model['type'] == 'yolo':
                    # YOLO models support both .pt and non-.pt formats
                    valid_models.append(model['model'])
                    if model['model'].endswith('.pt'):
                        valid_models.append(model['model'][:-3])  # Remove .pt suffix
                else:
                    # torchvision models use underscore format
                    valid_models.append(model['model'])
                    valid_models.append(model['model'].replace('_', '-'))
            return model_name in valid_models
        
        elif task == 'segmentation':
            valid_models = []
            for model in SEGMENTATION_MODELS.values():
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                valid_models.append(model_id)
            return model_name in valid_models
        
        return False
    
    def _validate_dataset_name(self, task, dataset_name):
        """Validate if dataset name is valid"""
        valid_datasets = {
            'classification': ['MNIST', 'CIFAR-10', 'ImageNet-Sample'],
            'detection': ['COCO-Sample', 'KITTI', 'Test-Images'],
            'segmentation': ['Cityscapes', 'Synthetic-Segmentation']
        }
        
        return dataset_name in valid_datasets.get(task, [])
    
    def args_to_config(self, args):
        """Convert command line arguments to configuration object"""
        # Auto-select device and record selection logic
        if args.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                device_choice_reason = "Auto-selected GPU (CUDA available)"
            else:
                device = 'cpu'
                device_choice_reason = "Auto-selected CPU (CUDA unavailable)"
        else:
            device = args.device
            if device == 'cpu':
                device_choice_reason = "User specified CPU"
            elif device == 'cuda:0':
                device_choice_reason = "User specified GPU"
            else:
                device_choice_reason = f"User specified device: {device}"
        
        # Log device selection information
        self.logger.info(f"Device selection: {device} ({device_choice_reason})")
        
        # Find model information
        model_info = self._find_model_info(args.task, args.model)
        
        config = {
            'device': device,
            'device_choice_reason': device_choice_reason,
            'task': args.task,
            'model_info': model_info,
            'dataset_name': args.dataset,
            'test_samples': args.samples,
            'batch_size': args.batch_size,
            'output_dir': args.output_dir,
            'plot': args.plot,
            'quiet': args.quiet
        }
        
        return config
    
    def _find_model_info(self, task, model_name):
        """Find model information based on model name"""
        # Check if it's a URL or local path
        if self._is_url(model_name) or self._is_local_path(model_name):
            # Create custom model info for paths/URLs
            return {
                'name': f'Custom Model: {os.path.basename(model_name)}',
                'model': model_name,
                'type': self._infer_model_framework(task, model_name),
                'custom_path': model_name  # Store the original path/URL
            }
        
        # Otherwise, look up in predefined models
        if task == 'classification':
            for model in CLASSIFICATION_MODELS.values():
                if model['model'] == model_name:
                    return model
        
        elif task == 'detection':
            for model in DETECTION_MODELS.values():
                if model['type'] == 'yolo':
                    # Support both yolov8n and yolov8n.pt formats
                    if model['model'] == model_name or model['model'] == f"{model_name}.pt":
                        return model
                    if model['model'].endswith('.pt') and model['model'][:-3] == model_name:
                        return model
                elif model['type'] == 'torchvision':
                    if model['model'] == model_name or model['model'].replace('_', '-') == model_name:
                        return model
        
        elif task == 'segmentation':
            for model in SEGMENTATION_MODELS.values():
                model_id = f"{model['model'].lower()}_{model['encoder'].replace('-', '_')}"
                if model_id == model_name:
                    return model
        
        return None
    
    def _infer_model_framework(self, task, model_path):
        """Infer model framework based on task and path"""
        if task == 'classification':
            return 'timm'
        elif task == 'detection':
            # Check filename for hints
            path_lower = model_path.lower()
            if 'yolo' in path_lower:
                return 'yolo'
            elif 'faster' in path_lower or 'rcnn' in path_lower or 'fcos' in path_lower:
                return 'torchvision'
            else:
                # Default to YOLO for detection
                return 'yolo'
        elif task == 'segmentation':
            return 'smp'
        
        return 'unknown'
    
    def print_config_summary(self, config):
        """Print configuration summary"""
        if not config.get('quiet', False):
            print("\n" + "="*60)
            print("Benchmark Test Configuration:")
            print("="*60)
            print(f"Computing Device: {config['device']} ({config.get('device_choice_reason', 'Unknown reason')})")
            if config['device'].startswith('cuda') and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU Info: {device_name} ({memory_gb:.1f}GB)")
            print(f"Task: {config['task']}")
            print(f"Model: {config['model_info']['name']}")
            if 'custom_path' in config['model_info']:
                print(f"Model Source: {config['model_info']['custom_path']}")
            print(f"Dataset: {config['dataset_name']}")
            print(f"Samples: {config['test_samples'] if config['test_samples'] != -1 else 'All'}")
            print(f"Output Directory: {config['output_dir']}")
            print(f"Generate Plots: {'Yes' if config['plot'] else 'No'}")
            print(f"Quiet Mode: {'Yes' if config['quiet'] else 'No'}")
            print("="*60)            
            print(f"Dataset: {config['dataset_name']}")
            print(f"Samples: {config['test_samples'] if config['test_samples'] != -1 else 'All'}")
            print(f"Output Directory: {config['output_dir']}")
            print(f"Generate Plots: {'Yes' if config['plot'] else 'No'}")
            print(f"Quiet Mode: {'Yes' if config['quiet'] else 'No'}")
            print("="*60)