#!/usr/bin/env python3
"""
Model module - responsible for loading and managing various deep learning models
"""

import os
import logging
import time
import torch
import torchvision.models.detection as detection_models
from pathlib import Path
from core.config import DETECTION_MODELS, CLASSIFICATION_MODELS, SEGMENTATION_MODELS
from core.utils import check_dependencies

# Check dependencies
dependencies = check_dependencies()

class ModelLoader:
    """Model loader class"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.dependencies = dependencies
    
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
    
    def _download_model_from_url(self, url):
        """Download model from URL"""
        self.logger.info(f"Downloading model from URL: {url}")
        print(f"Downloading model from URL: {url}")
        
        try:
            import urllib.request
            import tempfile
            
            # Create temporary directory for downloaded models
            temp_dir = Path(tempfile.gettempdir()) / "benchmark_models"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename or '.' not in filename:
                filename = 'downloaded_model.pth'
            
            local_path = temp_dir / filename
            
            # Download with progress
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\rDownloading: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, str(local_path), reporthook)
            print()  # New line after download
            
            self.logger.info(f"Model downloaded successfully to: {local_path}")
            print(f"Model downloaded to: {local_path}")
            
            return str(local_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download model from URL: {e}")
            raise RuntimeError(f"Failed to download model from {url}: {e}")
    
    def _resolve_model_path(self, model_identifier, model_type):
        """Resolve model identifier to actual path or name"""
        # Check if it's a URL
        if self._is_url(model_identifier):
            return self._download_model_from_url(model_identifier)
        
        # Check if it's a local path
        if self._is_local_path(model_identifier):
            # Expand user home directory (~)
            expanded_path = os.path.expanduser(model_identifier)
            abs_path = os.path.abspath(expanded_path)
            
            if os.path.exists(abs_path):
                self.logger.info(f"Using local model file: {abs_path}")
                print(f"Loading model from local path: {abs_path}")
                return abs_path
            else:
                raise FileNotFoundError(f"Model file not found: {abs_path}")
        
        # Otherwise, treat as model name
        return model_identifier
    
    def load_classification_model(self, model_info):
        """Load classification model"""
        model_identifier = model_info['model']
        resolved_path = self._resolve_model_path(model_identifier, 'classification')
        
        self.logger.info(f"Loading classification model: {resolved_path}")
        
        if not self.dependencies['timm']:
            raise ImportError("timm library not available. Install with: pip install timm")
        
        import timm
        
        # Check if it's a local file or URL-downloaded file
        if os.path.isfile(resolved_path):
            # Load from checkpoint
            self.logger.info(f"Loading classification model from checkpoint: {resolved_path}")
            print(f"Loading from checkpoint: {resolved_path}")
            
            try:
                # Try to load as timm model checkpoint
                model = timm.create_model('resnet18', pretrained=False, num_classes=1000)
                checkpoint = torch.load(resolved_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                model = model.to(self.device)
                self.logger.info("Classification model loaded from checkpoint successfully")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load as timm checkpoint: {e}, trying direct torch.load")
                # Try direct model load
                model = torch.load(resolved_path, map_location=self.device)
                if hasattr(model, 'eval'):
                    model.eval()
                return model
        else:
            # Load from timm model zoo by name
            model = timm.create_model(
                resolved_path, 
                pretrained=True,
                num_classes=1000
            )
            model.eval()
            model = model.to(self.device)
            
            self.logger.info(f"Classification model loaded successfully: {model_info['name']}")
            return model
    
    def load_detection_model(self, model_info):
        """Load detection model"""
        if model_info['type'] == 'yolo':
            return self._load_yolo_model(model_info)
        elif model_info['type'] == 'torchvision':
            return self._load_torchvision_detection_model(model_info)
        else:
            raise ValueError(f"Unsupported detection model type: {model_info['type']}")
    
    def _load_yolo_model(self, model_info):
        """Load YOLO model"""
        if not self.dependencies['ultralytics']:
            raise ImportError("ultralytics library not available. Install with: pip install ultralytics")
        
        from ultralytics import YOLO
        
        model_identifier = model_info['model']
        resolved_path = self._resolve_model_path(model_identifier, 'detection')
        
        self.logger.info(f"Loading YOLO detection model: {resolved_path}")
        
        model = YOLO(resolved_path)
        
        self.logger.info(f"YOLO model loaded successfully: {model_info['name']}")
        return model
    
    def _load_torchvision_detection_model(self, model_info):
        """Load torchvision detection model"""
        if not self.dependencies['torchvision_detection']:
            raise ImportError("torchvision detection models not available")
        
        model_identifier = model_info['model']
        resolved_path = self._resolve_model_path(model_identifier, 'detection')
        
        self.logger.info(f"Loading torchvision detection model: {resolved_path}")
        
        # Check if it's a local file
        if os.path.isfile(resolved_path):
            self.logger.info(f"Loading torchvision model from checkpoint: {resolved_path}")
            print(f"Loading from checkpoint: {resolved_path}")
            
            try:
                # Try loading checkpoint
                checkpoint = torch.load(resolved_path, map_location='cpu')
                
                # Create a default model structure
                model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
                
                # Load state dict
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                model = model.to(self.device)
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load as checkpoint: {e}, trying direct model load")
                model = torch.load(resolved_path, map_location=self.device)
                if hasattr(model, 'eval'):
                    model.eval()
                return model
        else:
            # Load from torchvision model zoo by name
            if resolved_path == 'fasterrcnn_resnet50_fpn':
                model = detection_models.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            elif resolved_path == 'fasterrcnn_mobilenet_v3_large_fpn':
                model = detection_models.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
            elif resolved_path == 'fcos_resnet50_fpn':
                model = detection_models.fcos_resnet50_fpn(weights='DEFAULT')
            else:
                raise ValueError(f"Unsupported torchvision detection model: {resolved_path}")
            
            model.eval()
            model = model.to(self.device)
            
            self.logger.info(f"Torchvision detection model loaded successfully: {model_info['name']}")
            return model
    
    def load_segmentation_model(self, model_info):
        """Load segmentation model"""
        if not self.dependencies['smp']:
            raise ImportError("segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")
        
        import segmentation_models_pytorch as smp
        
        model_identifier = model_info['model']
        # For segmentation, we need to check if it's a custom path
        # The model_info might have been constructed from the identifier
        
        # Check if we have a custom model path
        if 'custom_path' in model_info:
            resolved_path = self._resolve_model_path(model_info['custom_path'], 'segmentation')
            
            if os.path.isfile(resolved_path):
                self.logger.info(f"Loading segmentation model from checkpoint: {resolved_path}")
                print(f"Loading from checkpoint: {resolved_path}")
                
                try:
                    # Try loading checkpoint
                    checkpoint = torch.load(resolved_path, map_location='cpu')
                    
                    # Create a default model structure
                    model_class = getattr(smp, 'Unet')
                    model = model_class(
                        encoder_name='resnet34',
                        encoder_weights=None,
                        classes=19,
                        activation=None
                    )
                    
                    # Load state dict
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    model = model.to(self.device)
                    return model
                except Exception as e:
                    self.logger.warning(f"Failed to load as checkpoint: {e}, trying direct model load")
                    model = torch.load(resolved_path, map_location=self.device)
                    if hasattr(model, 'eval'):
                        model.eval()
                    return model
        
        # Otherwise, load from model zoo
        self.logger.info(f"Loading segmentation model using segmentation_models_pytorch: {model_info['model']}")
        
        # Create model using segmentation_models_pytorch
        model_class = getattr(smp, model_info['model'])
        model = model_class(
            encoder_name=model_info['encoder'],
            encoder_weights='imagenet',
            classes=19,  # Cityscapes has 19 classes
            activation=None
        )
        model.eval()
        model = model.to(self.device)
        
        self.logger.info(f"Segmentation model loaded successfully: {model_info['name']}")
        return model
    
    def load_model(self, model_type, model_info):
        """Load corresponding model based on model type"""
        self.logger.info(f"Starting to load model: {model_info['name']}")
        print(f"\nLoading model: {model_info['name']}...")
        
        try:
            if model_type == 'classification':
                model = self.load_classification_model(model_info)
            elif model_type == 'detection':
                model = self.load_detection_model(model_info)
            elif model_type == 'segmentation':
                model = self.load_segmentation_model(model_info)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.logger.info(f"Model loaded successfully: {model_info['name']}")
            print("Model loaded successfully!")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

def get_available_models(model_type, dependencies):
    """Get list of available models"""
    if model_type == 'classification':
        if dependencies['timm']:
            return CLASSIFICATION_MODELS
        else:
            return {}
    
    elif model_type == 'detection':
        available_models = {}
        for key, value in DETECTION_MODELS.items():
            if value['type'] == 'yolo' and dependencies['ultralytics']:
                available_models[key] = value
            elif value['type'] == 'torchvision' and dependencies['torchvision_detection']:
                available_models[key] = value
        return available_models
    
    elif model_type == 'segmentation':
        if dependencies['smp']:
            return SEGMENTATION_MODELS
        else:
            return {}
    
    return {}

def validate_model_availability(model_type, dependencies):
    """Validate if model type is available"""
    if model_type == 'classification':
        return dependencies['timm']
    elif model_type == 'detection':
        return dependencies['ultralytics'] or dependencies['torchvision_detection']
    elif model_type == 'segmentation':
        return dependencies['smp']
    return False
